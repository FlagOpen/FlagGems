# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for fbgemm_linear_int8_weight (delegated implementation).

Reference strategy
------------------
``fbgemm_linear_int8_weight`` is delegated to aten (there is no Triton
kernel). ``packed`` is an opaque CPU tensor whose TensorImpl carries an fbgemm
``PackBMatrix<int8_t>`` C++ object; the packed layout is owned by the fbgemm
library and cannot be synthesized by FlagGems. On this build the op only has a
CPU kernel, and the CUDA-resident path of native aten segfaults (the fbgemm
CPU SIMD unpacking dereferences device pointers), so the FlagGems wrapper
stages CUDA activations through host memory around the same aten call.

The tests therefore validate:

* the numerical contract against a **logical-equivalent, fbgemm-free
  reference**: the kernel dynamically quantizes the fp32 activations to uint8
  and dequantizes with the int8 weight ``(w - weight_zero_point) *
  weight_scale``, so the reference is
  ``torch.quantize_per_tensor(input, sx, szp, quint8).dequantize()`` composed
  with ``F.linear(x_q, (w.int8 - weight_zero_point) * weight_scale, bias)``
  (the same composition is the recipe used by PyTorch's own
  ``test_quantized_op.py`` for this op). The quantization grid of the two
  sides is computed independently (Python fp64 vs fbgemm fp32 arithmetic), so
  quantized values can differ by one unit at rounding boundaries; for large
  activations a rare boundary row can exceed the analytic
  ``K * sx * weight_scale`` bound, hence a 8x headroom on the tolerance;
* exact device routing: CUDA inputs flow through the wrapper (host staging)
  and are bitwise equal to feeding the same data to the same aten kernel on
  CPU - which is where native aten computes, so this is a parity check;
* the wrapper's validation logic (dtype / dim / error paths) via illegal
  inputs.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# fbgemm_linear_int8_weight accepts fp32 activations only (fp16/fp64/bf16 are
# rejected by the aten int8-weight kernel) and produces fp32 output, so the
# dtype matrix is fixed instead of parametrizing over FLOAT_DTYPES.
SHAPES = (
    [(16, 32), (64, 128)]
    if utils.QUICK_MODE
    else [
        (8, 16),
        (64, 128),
        (512, 512),
        (16, 128, 64),
    ]
)

# The packed weight must be built on CPU: the fbgemm pack kernel has no CUDA
# registration, and moving a packed tensor to CUDA (or cloning it) drops the
# PackBMatrix wrapper stored in the TensorImpl.
N_OUT, N_IN = 16, 32


def _quantized_weight(n_out=N_OUT, n_in=N_IN, seed=0):
    """Build (weight, packed, col_offsets, scale, zero_point) like the op expects.

    Follows the recipe of PyTorch's own test_quantized_op.py:
    quantize the fp32 weight per-tensor to int8 via
    ``fbgemm_linear_quantize_weight``, compute the per-column offsets and pack.
    """
    generator = torch.Generator().manual_seed(seed)
    weight_fp32 = torch.randn(n_out, n_in, generator=generator, dtype=torch.float32)
    weight_int8, col_offsets, weight_scale, weight_zero_point = (
        torch.fbgemm_linear_quantize_weight(weight_fp32)
    )
    packed = torch.ops.aten.fbgemm_pack_quantized_matrix(weight_int8)
    return (
        weight_fp32,
        weight_int8,
        packed,
        col_offsets,
        weight_scale,
        weight_zero_point,
    )


def _reference(input, weight_int8, weight_scale, weight_zero_point, bias):
    """fbgemm-free logical equivalent.

    The kernel quantizes the activations to uint8 with dynamically computed
    parameters (0..255 affine on the input min/max, computed by fbgemm on the
    host) and multiplies the dequantized activations by the dequantized int8
    weight ``(w - weight_zero_point) * weight_scale``.

    The quantization grid of the reference is derived from the **host** copy
    of the activations: the wrapper stages CUDA activations to the CPU kernel,
    and the fbgemm grid is a host-side fp32 computation whose zero-point
    rounding has been observed to differ by one unit from a CUDA-side
    ``torch.quantize_per_tensor`` on some torch builds. Deriving the grid from
    ``input.cpu()`` keeps the two grids consistent (the remaining difference
    is only fbgemm int-math vs torch fp32 linear accumulation).
    """
    host_input = input.cpu()
    x_min = host_input.min()
    x_max = host_input.max()
    x_scale = (x_max - x_min) / 255.0
    x_zp = torch.round(-x_min / x_scale)
    host_grid = torch.quantize_per_tensor(host_input, x_scale, x_zp, dtype=torch.quint8)
    weight_deq = (
        weight_int8.to(input.device).float() - weight_zero_point
    ) * weight_scale
    return torch.nn.functional.linear(
        host_grid.dequantize().to(input.device), weight_deq, bias
    )


@pytest.mark.fbgemm_linear_int8_weight
@pytest.mark.parametrize("shape", SHAPES)
def test_fbgemm_linear_int8_weight(shape):
    (
        weight_fp32,
        weight_int8,
        packed,
        col_offsets,
        weight_scale,
        weight_zero_point,
    ) = _quantized_weight(n_in=shape[-1])
    input = torch.randn(shape, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)

    res_out = flag_gems.fbgemm_linear_int8_weight(
        input, weight_int8, packed, col_offsets, weight_scale, weight_zero_point, bias
    )

    # The FlagGems wrapper executes the aten fbgemm CPU kernel (host staging
    # for CUDA activations); the logical reference replicates the uint8
    # activation quantization + int8 weight dequantization in pure torch on
    # the device. The two sides compute the quantization grid independently
    # (Python fp64 vs fbgemm fp32), so quantized activations can differ by one
    # unit at rounding boundaries; the analytic bound is
    # (K + 1) * x_scale * weight_scale, but a rare boundary row can exceed it
    # (fbgemm SIMD edge case), hence the 8x headroom.
    ref_out = _reference(
        utils.to_reference(input),
        weight_int8,
        weight_scale,
        weight_zero_point,
        utils.to_reference(bias),
    )
    k = shape[-1]
    x_scale = (input.max() - input.min()).item() / 255.0
    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype=torch.float32,
        atol=8 * (k + 1) * x_scale * weight_scale + 1e-5,
    )


@pytest.mark.fbgemm_linear_int8_weight
def test_fbgemm_linear_int8_weight_matches_cpu_kernel():
    """CUDA inputs must be bitwise equal to the same aten kernel fed on CPU.

    The wrapper stages CUDA activations to host memory and calls the same
    fbgemm CPU kernel, so with identical data the two calls execute the exact
    same integer arithmetic: the results are bitwise equal.
    """
    _, weight_int8, packed, col_offsets, weight_scale, weight_zero_point = (
        _quantized_weight(n_in=64, seed=3)
    )
    input = torch.randn(16, 128, 64, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)

    res_out = flag_gems.fbgemm_linear_int8_weight(
        input, weight_int8, packed, col_offsets, weight_scale, weight_zero_point, bias
    )
    ref_out = flag_gems.fbgemm_linear_int8_weight(
        input.cpu(),
        weight_int8,
        packed,
        col_offsets,
        weight_scale,
        weight_zero_point,
        bias.cpu(),
    )
    # The wrapper stages CUDA activations to the CPU kernel and stages the
    # result back to the input device; the CPU-side call runs the same kernel
    # with the same data, so the two results are bitwise equal once both are
    # compared on CPU (to_reference only moves when --ref cpu is set).
    utils.gems_assert_equal(utils.to_reference(res_out).cpu(), ref_out.cpu())


@pytest.mark.fbgemm_linear_int8_weight
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.bfloat16])
def test_fbgemm_linear_int8_weight_unsupported_activation(dtype):
    """Non-fp32 activations are rejected before dispatch, like aten."""
    _, weight_int8, packed, col_offsets, weight_scale, weight_zero_point = (
        _quantized_weight()
    )
    input = torch.randn(4, N_IN, device=flag_gems.device, dtype=dtype)
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)
    with pytest.raises(RuntimeError, match="expected scalar type Float"):
        flag_gems.fbgemm_linear_int8_weight(
            input,
            weight_int8,
            packed,
            col_offsets,
            weight_scale,
            weight_zero_point,
            bias,
        )


@pytest.mark.fbgemm_linear_int8_weight
def test_fbgemm_linear_int8_weight_rejects_1d_input():
    """A vector input has no matrix dimension: rejected like aten."""
    _, weight_int8, packed, col_offsets, weight_scale, weight_zero_point = (
        _quantized_weight()
    )
    input = torch.randn(N_IN, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)
    with pytest.raises(RuntimeError, match=r"input\.dim\(\) >= 2"):
        flag_gems.fbgemm_linear_int8_weight(
            input,
            weight_int8,
            packed,
            col_offsets,
            weight_scale,
            weight_zero_point,
            bias,
        )


@pytest.mark.fbgemm_linear_int8_weight
def test_fbgemm_linear_int8_weight_k_mismatch():
    """An input whose last dim differs from the weight K raises."""
    _, weight_int8, packed, col_offsets, weight_scale, weight_zero_point = (
        _quantized_weight()
    )
    input = torch.randn(8, N_IN + 4, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)
    with pytest.raises(RuntimeError, match=r"weight\.size\(1\)"):
        flag_gems.fbgemm_linear_int8_weight(
            input,
            weight_int8,
            packed,
            col_offsets,
            weight_scale,
            weight_zero_point,
            bias,
        )


@pytest.mark.fbgemm_linear_int8_weight
def test_fbgemm_linear_int8_weight_unsupported_weight_dtype():
    """A non-int8 weight is rejected before dispatch (aten only checks dim/K)."""
    weight_fp32, _, packed, col_offsets, weight_scale, weight_zero_point = (
        _quantized_weight()
    )
    input = torch.randn(4, N_IN, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight_fp32.shape[0], device=flag_gems.device)
    with pytest.raises(RuntimeError, match="weight must have dtype Int8"):
        flag_gems.fbgemm_linear_int8_weight(
            input,
            weight_fp32,
            packed,
            col_offsets,
            weight_scale,
            weight_zero_point,
            bias,
        )


@pytest.mark.fbgemm_linear_int8_weight
def test_fbgemm_linear_int8_weight_unsupported_col_offsets_dtype():
    """A non-int32 col_offsets is rejected before dispatch."""
    _, weight_int8, packed, _, weight_scale, weight_zero_point = _quantized_weight()
    input = torch.randn(4, N_IN, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)
    col_offsets = weight_int8.sum(0, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="col_offsets must have dtype Int"):
        flag_gems.fbgemm_linear_int8_weight(
            input,
            weight_int8,
            packed,
            col_offsets,
            weight_scale,
            weight_zero_point,
            bias,
        )


@pytest.mark.fbgemm_linear_int8_weight
def test_fbgemm_linear_int8_weight_non_contiguous_input():
    """A strided activation is accepted and matches the contiguous result."""
    _, weight_int8, packed, col_offsets, weight_scale, weight_zero_point = (
        _quantized_weight()
    )
    pool = torch.randn(8, N_IN * 2, device=flag_gems.device, dtype=torch.float32)
    input = pool[:, ::2]
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)

    res_out = flag_gems.fbgemm_linear_int8_weight(
        input, weight_int8, packed, col_offsets, weight_scale, weight_zero_point, bias
    )
    ref_out = flag_gems.fbgemm_linear_int8_weight(
        input.contiguous(),
        weight_int8,
        packed,
        col_offsets,
        weight_scale,
        weight_zero_point,
        bias,
    )
    # Both sides come from the same wrapper; route through to_reference so the
    # --ref cpu device convention holds for the shared assert helper.
    utils.gems_assert_equal(utils.to_reference(res_out), utils.to_reference(ref_out))
