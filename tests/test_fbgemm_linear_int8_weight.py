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
  and multiplies by the dequantized int8 weight ``(w - weight_zero_point) *
  weight_scale``. The fbgemm quantization parameters are computed with
  different arithmetic than a torch-side ``quantize_per_tensor`` grid and can
  drift by one step on some builds, which would move every activation-weight
  product by one step. To make the comparison well-defined regardless of that
  rounding, the test feeds **non-negative activations that already sit on a
  uint8 grid point**: both grids clamp the minimum to code 0 (zero-point 0)
  and re-quantizing grid values is stable, so both sides multiply the same
  integer codes. The residual difference is fbgemm integer accumulation vs a
  torch fp32 linear reduction, bounded by ``(K + 1) * sx * weight_scale``.
  The composition itself follows the recipe used by PyTorch's own
  ``test_quantized_op.py`` for this op;
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


def _grid_exact_input(shape, dtype=torch.float32):
    """Non-negative activations that already sit on a uint8 quantization grid.

    The kernel quantizes the activations with dynamically computed parameters
    (0..255 affine on the input min/max, computed by fbgemm on the host). The
    fbgemm parameters use different arithmetic than a torch-side
    ``quantize_per_tensor`` grid and can drift by one step on some builds,
    which would move every activation-weight product by one step.

    Using non-negative values pins the zero-point of *both* grids to 0 (both
    clamp the minimum to code 0), and feeding values that already sit on a
    grid point keeps the integer codes stable under the remaining scale
    rounding: any near-identical grid maps them back to the same integer. The
    residual difference is then only fbgemm integer accumulation vs a torch
    fp32 linear reduction.
    """
    activations = torch.rand(shape, dtype=dtype)
    x_min = activations.min()
    x_max = activations.max()
    x_scale = (x_max - x_min) / 255.0
    quantized = torch.quantize_per_tensor(activations, x_scale, 0, dtype=torch.quint8)
    return quantized.dequantize(), x_scale.item()


def _reference(input, weight_int8, weight_scale, weight_zero_point, bias):
    """fbgemm-free logical equivalent.

    ``input`` is already on the quantization grid (see :func:`_grid_exact`),
    so the reference is simply the linear layer with the dequantized int8
    weight ``(w - weight_zero_point) * weight_scale``.
    """
    weight_deq = (
        weight_int8.to(input.device).float() - weight_zero_point
    ) * weight_scale
    if bias is not None:
        bias = bias.to(input.device)
    return torch.nn.functional.linear(input, weight_deq, bias)


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
    raw_input, x_scale = _grid_exact_input(shape)
    input = raw_input.to(flag_gems.device)
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)

    res_out = flag_gems.fbgemm_linear_int8_weight(
        input, weight_int8, packed, col_offsets, weight_scale, weight_zero_point, bias
    )

    # The FlagGems wrapper executes the aten fbgemm CPU kernel (host staging
    # for CUDA activations); the logical reference replicates the int8 weight
    # dequantization with a plain fp32 linear layer. Both sides multiply the
    # same uint8-grid activations (zero-point 0, see _grid_exact_input) by the
    # same dequantized weight; only the integer accumulation order vs fp32
    # reduction differs, bounded by (K + 1) * x_scale * weight_scale (with 4x
    # headroom for reduction noise).
    ref_out = _reference(
        utils.to_reference(input),
        weight_int8,
        weight_scale,
        weight_zero_point,
        utils.to_reference(bias),
    )
    k = shape[-1]
    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype=torch.float32,
        atol=4 * (k + 1) * x_scale * weight_scale + 1e-5,
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
