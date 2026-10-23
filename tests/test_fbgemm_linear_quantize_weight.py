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
"""Tests for fbgemm_linear_quantize_weight (delegated implementation).

Reference strategy
------------------
``fbgemm_linear_quantize_weight`` is delegated to aten (there is no Triton
kernel). The only registered backend is the fbgemm CPU SIMD code, whose
quantization grid comes from ``fbgemm::ChooseQuantizationParams`` (fp32 SIMD
arithmetic over a global min/max pass). Reimplementing that grid in a Python
reference is exactly the trap that failed PR #7638 in CI: the fbgemm-side
fp32 rounding boundaries and a Python fp64 reference grid disagree on a small
fraction of elements (round-to-nearest ties, NaN/Inf clamping edges), and no
fixed analytic tolerance absorbs that across torch builds.

This test file therefore uses **structure + round-trip validation only**:

* the 4-tuple contract is checked structurally: ``weight_q`` is int8 with the
  input's shape, ``col_offsets`` is int32 with shape ``(input.size(0),)``,
  ``scale`` is a float and ``zero_point`` is an int;
* the values are validated through the quantization **round-trip bound**
  ``dequant(w) = (weight_q - zero_point) * scale`` (element-wise), whose error
  against the input is bounded by ``scale / 2`` on every build by
  construction - a property of the affine grid itself, independent of how
  fbgemm picked the grid parameters;
* ``col_offsets`` are validated against the identity that holds for the
  canonical 2-D use case: ``col_offsets == (weight_q - zero_point).sum(1)``;
* the CUDA staging path must produce results bitwise-equal to the CPU
  delegation of the same weight (both sides run the identical aten kernel);
* the error paths (non-fp32 dtype, 0-D input) mirror aten rejections.

Nothing here compares fbgemm numerics against a re-derived reference, so CI
never depends on fbgemm-vs-reference rounding differences.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# fbgemm_linear_quantize_weight accepts fp32 weights only (fp16/fp64/bf16 are
# rejected by the aten kernel) and returns (int8, int32, float, int), so the
# dtype matrix is fixed instead of parametrizing over FLOAT_DTYPES.
SHAPES = (
    [(16, 32), (64, 128)]
    if utils.QUICK_MODE
    else [
        (8, 16),
        (64, 128),
        (512, 512),
        (16, 32, 64),
    ]
)


def _assert_quantize_contract(weight, result):
    """Structural + round-trip validation of the 4-tuple contract.

    weight_q is int8 with the input's shape, col_offsets is int32 with shape
    (weight.size(0),), scale is a float and zero_point is an int. The
    dequantized weight must lie within scale/2 of the input everywhere (the
    round-to-nearest bound of the affine grid), and col_offsets must equal the
    per-row sums of (weight_q - zero_point) as produced by the fbgemm
    CalcColOffsetsTranspose helper for the canonical 2-D case.
    """
    weight_q, col_offsets, scale, zero_point = result

    # Structural contract (schema: Tensor, Tensor, float, int).
    assert isinstance(weight_q, torch.Tensor)
    assert weight_q.dtype == torch.int8
    assert weight_q.shape == weight.shape
    assert weight_q.device.type == weight.device.type
    assert isinstance(col_offsets, torch.Tensor)
    assert col_offsets.dtype == torch.int32
    assert col_offsets.shape == (weight.shape[0],)
    assert col_offsets.device.type == weight.device.type
    assert isinstance(scale, float)
    assert isinstance(zero_point, int)

    # Round-trip bound: q = clamp(round(w/s) + zp, -128, 127) implies
    # |(q - zp) * s - w| <= s/2 wherever the value is not clamped, and the
    # grid is chosen so clamping only affects values outside [w_min, w_max].
    # The dequantize multiplication itself rounds to fp32, adding a relative
    # eps to the bound; 1e-4 absorbs it with wide margin (observed worst
    # overshoot is ~7e-6 relative).
    dequantized = (weight_q.to(torch.float32) - zero_point) * scale
    err = (dequantized - weight).abs()
    assert bool((err <= scale / 2 * (1 + 1e-4)).all())

    # The quantized values must lie in the int8 affine grid.
    assert weight_q.min().item() >= -128
    assert weight_q.max().item() <= 127

    # 2-D col_offsets identity (fbgemm CalcColOffsetsTranspose on a matrix:
    # col_offsets[i] is the sum of row i of the quantized weight minus
    # zero_point per element).
    if weight.dim() == 2:
        expected = (weight_q.to(torch.int64) - zero_point).sum(1, dtype=torch.int32)
        assert torch.equal(col_offsets, expected)


@pytest.mark.fbgemm_linear_quantize_weight
@pytest.mark.parametrize("shape", SHAPES)
def test_fbgemm_linear_quantize_weight_round_trip(shape):
    """Quantize an fp32 weight and validate the 4-tuple contract.

    On CPU inputs the wrapper calls the aten fbgemm kernel directly; on CUDA
    inputs it stages the weight through host memory around the same call and
    moves weight_q / col_offsets back to the input's device (scale and
    zero_point are host scalars). Values are validated via the round-trip
    error bound of the affine grid, never against a re-derived reference grid
    (see module docstring).
    """
    weight = torch.randn(shape, device=flag_gems.device, dtype=torch.float32)
    result = flag_gems.fbgemm_linear_quantize_weight(weight)
    _assert_quantize_contract(weight, result)


@pytest.mark.fbgemm_linear_quantize_weight
def test_fbgemm_linear_quantize_weight_cuda_matches_cpu():
    """The CUDA staging path must match the CPU delegation bitwise.

    Both sides run the same aten fbgemm CPU kernel on the same host payload,
    so this stays a bitwise check against the delegated kernel itself (no
    cross-implementation reference).
    """
    if torch.cuda.is_available():
        target_device = torch.device(flag_gems.device)
    else:
        pytest.skip("CUDA staging path requires a CUDA device")
    weight_cpu = torch.randn(32, 64, dtype=torch.float32)
    weight_cuda = weight_cpu.cuda(target_device)

    cpu_q, cpu_offsets, cpu_scale, cpu_zp = flag_gems.fbgemm_linear_quantize_weight(
        weight_cpu
    )
    cuda_q, cuda_offsets, cuda_scale, cuda_zp = flag_gems.fbgemm_linear_quantize_weight(
        weight_cuda
    )
    assert torch.equal(cpu_q.cuda(target_device), cuda_q)
    assert torch.equal(cpu_offsets.cuda(target_device), cuda_offsets)
    assert cpu_scale == cuda_scale
    assert cpu_zp == cuda_zp


@pytest.mark.fbgemm_linear_quantize_weight
def test_fbgemm_linear_quantize_weight_non_contiguous_input():
    """A strided weight is accepted and matches the contiguous result."""
    pool = torch.randn(16, 64, device=flag_gems.device, dtype=torch.float32)
    weight = pool[:, ::2]

    res = flag_gems.fbgemm_linear_quantize_weight(weight)
    ref = flag_gems.fbgemm_linear_quantize_weight(weight.contiguous())
    # Both sides come from the same wrapper; route through to_reference so the
    # --ref cpu device convention holds for the shared assert helper.
    utils.gems_assert_equal(utils.to_reference(res[0]), utils.to_reference(ref[0]))
    utils.gems_assert_equal(utils.to_reference(res[1]), utils.to_reference(ref[1]))
    assert res[2] == ref[2]
    assert res[3] == ref[3]


@pytest.mark.fbgemm_linear_quantize_weight
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.bfloat16])
def test_fbgemm_linear_quantize_weight_unsupported_dtype(dtype):
    """Non-fp32 weights are rejected before dispatch, like aten."""
    weight = torch.randn(8, 16, device=flag_gems.device, dtype=dtype)
    with pytest.raises(RuntimeError, match="expected scalar type Float"):
        flag_gems.fbgemm_linear_quantize_weight(weight)


@pytest.mark.fbgemm_linear_quantize_weight
def test_fbgemm_linear_quantize_weight_rejects_0d_input():
    """A 0-D weight has no row dimension: rejected like aten (which dies
    with an IndexError on the trailing dimension)."""
    weight = torch.randn((), device=flag_gems.device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match=r"input\.dim\(\) >= 1"):
        flag_gems.fbgemm_linear_quantize_weight(weight)


@pytest.mark.fbgemm_linear_quantize_weight
def test_fbgemm_linear_quantize_weight_rejects_non_tensor():
    """A non-tensor input is rejected with a TypeError."""
    with pytest.raises(TypeError, match="input must be a torch.Tensor"):
        flag_gems.fbgemm_linear_quantize_weight([[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.fbgemm_linear_quantize_weight
def test_fbgemm_linear_quantize_weight_empty_rows():
    """A 0-row weight quantizes to empty outputs with a valid grid."""
    weight = torch.randn(0, 16, device=flag_gems.device, dtype=torch.float32)
    weight_q, col_offsets, scale, zero_point = flag_gems.fbgemm_linear_quantize_weight(
        weight
    )
    assert weight_q.dtype == torch.int8
    assert weight_q.shape == (0, 16)
    assert col_offsets.dtype == torch.int32
    assert col_offsets.shape == (0,)
    assert isinstance(scale, float)
    assert isinstance(zero_point, int)
