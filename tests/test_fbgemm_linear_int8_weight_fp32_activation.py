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
"""Tests for fbgemm_linear_int8_weight_fp32_activation (delegated implementation).

Reference strategy
------------------
``fbgemm_linear_int8_weight_fp32_activation`` is delegated to aten (there is
no Triton kernel). ``packed`` is an opaque CPU tensor whose TensorImpl carries
an fbgemm ``PackBMatrix<int8_t>`` C++ object; the packed layout is owned by
the fbgemm library and cannot be synthesized by FlagGems.

**Deliberately no end-to-end numerical comparison.** The sibling op
``fbgemm_linear_int8_weight`` (PR #7638) validated numerically against a
fbgemm-free logical reference (``torch.quantize_per_tensor`` activation
quantization composed with ``F.linear(x_q, (w - zp) * scale, bias)``). That
passed on a local torch build but failed in CI, where the torch build ships
the real fbgemm library: the fbgemm-side quantization grid (fbgemm SIMD fp32
arithmetic) and the Python fp64 reference grid disagree at rounding
boundaries, producing element differences that no fixed analytic tolerance
absorbs across builds. The same applies to a dequantized-weight reference
(``F.linear(x, (w - zp) * scale, bias)``) and to ``fbgemm_linear_prepack``
unpacking. This test file therefore restricts itself to behavior that is
identical on every build (validation and error paths, documented deprecation
behavior) plus a delegation-path smoke check guarded by
``pytest.importorskip``-style availability probing, so CI never depends on
fbgemm-vs-reference numerics.

Validation covers:

* non-fp32 activations / non-int8 weight / non-int32 col_offsets are rejected
  before dispatch (the aten kernel itself only checks dim/K);
* 1-D input rejection (matrix-only op, same message as aten);
* K mismatch rejection (same message as aten);
* scalar type validation of weight_scale / weight_zero_point;
* fp32-only bias.

The successful delegation path (a legal call must not raise) is exercised
only when the environment can actually build the fbgemm packed object; where
that is unavailable or unsafe the test is skipped rather than weakened.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# fbgemm_linear_int8_weight_fp32_activation accepts fp32 activations only
# (fp16/fp64/bf16 are rejected by the aten kernel) and produces fp32 output,
# so the dtype matrix is fixed instead of parametrizing over FLOAT_DTYPES.
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

# The packed object / quantize helpers come from the fbgemm-backed aten CPU
# kernels. They are deprecated on recent builds and their availability varies
# across torch builds, so every test that needs them skips when absent
# (probed lazily via torch.ops so the module itself imports cleanly).


@pytest.fixture(autouse=True)
def _require_fbgemm_helpers():
    global _fbgemm_pack_quantized_matrix, _fbgemm_linear_quantize_weight
    if not hasattr(torch.ops.aten, "fbgemm_pack_quantized_matrix") or not hasattr(
        torch, "fbgemm_linear_quantize_weight"
    ):
        pytest.skip(
            "fbgemm-backed aten helpers (fbgemm_pack_quantized_matrix / "
            "fbgemm_linear_quantize_weight) unavailable on this torch build"
        )
    _fbgemm_pack_quantized_matrix = torch.ops.aten.fbgemm_pack_quantized_matrix
    _fbgemm_linear_quantize_weight = torch.fbgemm_linear_quantize_weight


def _quantized_weight(n_out=N_OUT, n_in=N_IN, seed=0):
    """Build (weight_fp32, weight_int8, packed, col_offsets, scale, zero_point)."""
    generator = torch.Generator().manual_seed(seed)
    weight_fp32 = torch.randn(n_out, n_in, generator=generator, dtype=torch.float32)
    weight_int8, col_offsets, weight_scale, weight_zero_point = (
        _fbgemm_linear_quantize_weight(weight_fp32)
    )
    packed = _fbgemm_pack_quantized_matrix(weight_int8)
    return (
        weight_fp32,
        weight_int8,
        packed,
        col_offsets,
        weight_scale,
        weight_zero_point,
    )


def _assert_delegates(input, weight_int8, packed, col_offsets, scale, zp, bias):
    """A legal call must return an fp32 result of the right shape, on the
    input's device, without raising. No numerical assertion: the delegated
    aten fbgemm kernel is the single source of truth, and its quantized
    arithmetic differs across torch builds (see module docstring)."""
    res_out = flag_gems.fbgemm_linear_int8_weight_fp32_activation(
        input, weight_int8, packed, col_offsets, scale, zp, bias
    )
    assert res_out.dtype == torch.float32
    assert res_out.shape == input.shape[:-1] + (weight_int8.shape[0],)
    assert res_out.device.type == input.device.type


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("shape", SHAPES)
def test_fbgemm_linear_int8_weight_fp32_activation_delegates(shape):
    """A legal call delegates to the aten fbgemm kernel without raising.

    On CPU inputs the wrapper calls the aten kernel directly; on CUDA inputs
    it stages activations/bias through host memory around the same call. The
    packed object itself must stay host-resident: the fbgemm pack kernel has
    no CUDA registration and moving/cloning a packed tensor drops the
    PackBMatrix C++ wrapper it carries. Output is fp32 with shape
    (..., N_out); values come verbatim from the delegated kernel (no
    numerical comparison - see module docstring).
    """
    _, weight_int8, packed, col_offsets, weight_scale, weight_zero_point = (
        _quantized_weight(n_in=shape[-1])
    )
    input = torch.randn(shape, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)

    _assert_delegates(
        input,
        weight_int8,
        packed,
        col_offsets,
        weight_scale,
        weight_zero_point,
        bias,
    )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.bfloat16])
def test_fbgemm_linear_int8_weight_fp32_activation_unsupported_activation(dtype):
    """Non-fp32 activations are rejected before dispatch, like aten."""
    _, weight_int8, packed, col_offsets, weight_scale, weight_zero_point = (
        _quantized_weight()
    )
    input = torch.randn(4, N_IN, device=flag_gems.device, dtype=dtype)
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)
    with pytest.raises(RuntimeError, match="expected scalar type Float"):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            input,
            weight_int8,
            packed,
            col_offsets,
            weight_scale,
            weight_zero_point,
            bias,
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
def test_fbgemm_linear_int8_weight_fp32_activation_rejects_1d_input():
    """A vector input has no matrix dimension: rejected like aten."""
    _, weight_int8, packed, col_offsets, weight_scale, weight_zero_point = (
        _quantized_weight()
    )
    input = torch.randn(N_IN, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)
    with pytest.raises(RuntimeError, match=r"input\.dim\(\) >= 2"):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            input,
            weight_int8,
            packed,
            col_offsets,
            weight_scale,
            weight_zero_point,
            bias,
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
def test_fbgemm_linear_int8_weight_fp32_activation_k_mismatch():
    """An input whose last dim differs from the weight K raises."""
    _, weight_int8, packed, col_offsets, weight_scale, weight_zero_point = (
        _quantized_weight()
    )
    input = torch.randn(8, N_IN + 4, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)
    with pytest.raises(RuntimeError, match=r"weight\.size\(1\)"):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            input,
            weight_int8,
            packed,
            col_offsets,
            weight_scale,
            weight_zero_point,
            bias,
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
def test_fbgemm_linear_int8_weight_fp32_activation_unsupported_weight_dtype():
    """A non-int8 weight is rejected before dispatch (aten only checks dim/K)."""
    weight_fp32, _, packed, col_offsets, weight_scale, weight_zero_point = (
        _quantized_weight()
    )
    input = torch.randn(4, N_IN, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight_fp32.shape[0], device=flag_gems.device)
    with pytest.raises(RuntimeError, match="weight must have dtype Int8"):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            input,
            weight_fp32,
            packed,
            col_offsets,
            weight_scale,
            weight_zero_point,
            bias,
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
def test_fbgemm_linear_int8_weight_fp32_activation_unsupported_col_offsets_dtype():
    """A non-int32 col_offsets is rejected before dispatch."""
    _, weight_int8, packed, _, weight_scale, weight_zero_point = _quantized_weight()
    input = torch.randn(4, N_IN, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)
    col_offsets = weight_int8.sum(0, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="col_offsets must have dtype Int"):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            input,
            weight_int8,
            packed,
            col_offsets,
            weight_scale,
            weight_zero_point,
            bias,
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
def test_fbgemm_linear_int8_weight_fp32_activation_bad_weight_scale():
    """A non-float weight_scale is rejected before dispatch (aten:
    TORCH_CHECK(weight_scale.isFloatingPoint()))."""
    _, weight_int8, packed, col_offsets, _, weight_zero_point = _quantized_weight()
    input = torch.randn(4, N_IN, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)
    with pytest.raises(RuntimeError, match="weight_scale must be a floating point"):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            input,
            weight_int8,
            packed,
            col_offsets,
            2,
            weight_zero_point,
            bias,
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
def test_fbgemm_linear_int8_weight_fp32_activation_bad_weight_zero_point():
    """A non-integral weight_zero_point is rejected before dispatch (aten:
    TORCH_CHECK(weight_zero_point.isIntegral(false)))."""
    _, weight_int8, packed, col_offsets, weight_scale, _ = _quantized_weight()
    input = torch.randn(4, N_IN, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)
    with pytest.raises(RuntimeError, match="weight_zero_point must be an integral"):
        flag_gems.fbgemm_linear_int8_weight_fp32_activation(
            input,
            weight_int8,
            packed,
            col_offsets,
            weight_scale,
            0.5,
            bias,
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
def test_fbgemm_linear_int8_weight_fp32_activation_unsupported_bias_dtype():
    """A non-fp32 bias is rejected before dispatch (aten: "expected scalar
    type Float but found Half/BFloat16/Int")."""
    _, weight_int8, packed, col_offsets, weight_scale, weight_zero_point = (
        _quantized_weight()
    )
    input = torch.randn(4, N_IN, device=flag_gems.device, dtype=torch.float32)
    for bias_dtype in (torch.float16, torch.bfloat16, torch.int32):
        bias = torch.randn(weight_int8.shape[0], device=flag_gems.device).to(bias_dtype)
        with pytest.raises(RuntimeError, match="bias must have dtype Float"):
            flag_gems.fbgemm_linear_int8_weight_fp32_activation(
                input,
                weight_int8,
                packed,
                col_offsets,
                weight_scale,
                weight_zero_point,
                bias,
            )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
def test_fbgemm_linear_int8_weight_fp32_activation_non_contiguous_input():
    """A strided activation is accepted and equals the contiguous result.

    Both sides go through the same wrapper, so this stays a bitwise check
    against the delegated kernel itself (no cross-implementation reference).
    """
    _, weight_int8, packed, col_offsets, weight_scale, weight_zero_point = (
        _quantized_weight()
    )
    pool = torch.randn(8, N_IN * 2, device=flag_gems.device, dtype=torch.float32)
    input = pool[:, ::2]
    bias = torch.randn(weight_int8.shape[0], device=flag_gems.device)

    res_out = flag_gems.fbgemm_linear_int8_weight_fp32_activation(
        input, weight_int8, packed, col_offsets, weight_scale, weight_zero_point, bias
    )
    ref_out = flag_gems.fbgemm_linear_int8_weight_fp32_activation(
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
