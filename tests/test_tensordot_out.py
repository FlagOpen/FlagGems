# Copyright 2026 FlagOS Contributors
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

import pytest

import torch

import flag_gems

from . import accuracy_utils as utils

TENSORDOT_CASES = [
    # Classic dims=2 style contraction (last d of a, first d of b).
    ((3, 4, 5), (4, 5, 6), [1, 2], [0, 1]),
    # Single contracted dim -> plain matmul-like.
    ((16, 32), (32, 24), [1], [0]),
    # Reordered / non-adjacent contracted dims.
    ((3, 5, 4, 6), (6, 4, 5, 3), [2, 1, 3], [1, 2, 0]),
    # Negative dim indices.
    ((8, 7, 9), (9, 7, 5), [-1, 1], [0, 1]),
    # Larger contraction dimension.
    ((64, 128), (128, 96), [1], [0]),
    # Outer product (no contracted dims).
    ((2, 3), (4, 5), [], []),
    # Zero-sized free dimensions.
    ((0, 3, 4), (4, 5), [2], [0]),
    ((3, 4), (4, 0, 5), [1], [0]),
    # Zero-sized contracted dimensions.
    ((3, 0), (0, 5), [1], [0]),
    # Size-1 contracted dim on self -> broadcast against other (ATen contract).
    ((3, 1), (5, 6), [1], [0]),
    # Size-1 contracted dim on other -> broadcast against self.
    ((3, 5), (1, 6), [1], [0]),
    # Size-1 contracted dim on both sides.
    ((3, 1), (1, 6), [1], [0]),
    # Mixed: one contracted dim size-1 on self, another matching.
    ((2, 1, 4), (1, 3, 4), [1], [1]),
]

def _reduce_dim(shape_a, shape_b, dims_a, dims_b):
    # Effective reduction length is the broadcast contracted size (max of the
    # two per contracted axis pair), used to size the accuracy tolerance.
    na = len(shape_a)
    nb = len(shape_b)
    k = 1
    for da, db in zip(dims_a, dims_b):
        k *= max(shape_a[da % na], shape_b[db % nb])
    return max(k, 1)

@pytest.mark.tensordot_out
@pytest.mark.parametrize("shape_a, shape_b, dims_a, dims_b", TENSORDOT_CASES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_tensordot_out(shape_a, shape_b, dims_a, dims_b, dtype):
    torch.manual_seed(0)
    a = torch.randn(shape_a, dtype=dtype, device=flag_gems.device)
    b = torch.randn(shape_b, dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)

    # Drive the reference through the native out= path, not functional tensordot.
    ref_out = torch.empty(0, dtype=ref_a.dtype, device=ref_a.device)
    torch.tensordot(ref_a, ref_b, dims=(dims_a, dims_b), out=ref_out)

    out = torch.empty(0, dtype=dtype, device=flag_gems.device)
    res = flag_gems.tensordot_out(a, b, dims_a, dims_b, out=out)
    assert res is out, "tensordot_out should return the same out tensor"
    assert tuple(out.shape) == tuple(ref_out.shape)

    utils.gems_assert_close(
        out, ref_out, dtype, reduce_dim=_reduce_dim(shape_a, shape_b, dims_a, dims_b)
    )

@pytest.mark.tensordot_out
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_tensordot_out_contract(dtype):
    torch.manual_seed(0)
    a = torch.randn(3, 4, dtype=dtype, device=flag_gems.device)
    b = torch.randn(4, 5, dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)

    def ref_via_out(out_shape, non_contiguous=False):
        # Build a reference out tensor and drive it through torch's native out=
        # path so we validate against ATen's out= contract, not functional call.
        if non_contiguous:
            big = torch.empty(
                (out_shape[0] * 2,) + tuple(out_shape[1:]),
                dtype=ref_a.dtype,
                device=ref_a.device,
            )
            ro = big[::2]
        else:
            ro = torch.empty(out_shape, dtype=ref_a.dtype, device=ref_a.device)
        torch.tensordot(ref_a, ref_b, dims=([1], [0]), out=ro)
        return ro

    # Correctly sized out.
    ref_out = ref_via_out((3, 5))
    out = torch.empty(3, 5, dtype=dtype, device=flag_gems.device)
    out_pointer = out.data_ptr()
    result = flag_gems.tensordot_out(a, b, [1], [0], out=out)
    assert result is out
    assert out.data_ptr() == out_pointer
    assert tuple(out.shape) == tuple(ref_out.shape)
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=4)

    # Empty out -> resized to the required shape (no warning).
    ref_out = ref_via_out((0,))
    out_empty = torch.empty(0, dtype=dtype, device=flag_gems.device)
    result = flag_gems.tensordot_out(a, b, [1], [0], out=out_empty)
    assert result is out_empty
    assert tuple(out_empty.shape) == tuple(ref_out.shape) == (3, 5)
    utils.gems_assert_close(out_empty, ref_out, dtype, reduce_dim=4)

    # Non-contiguous out (a strided view) is written in place.
    ref_out = ref_via_out((3, 5), non_contiguous=True)
    big = torch.empty(6, 5, dtype=dtype, device=flag_gems.device)
    out_view = big[::2]
    out_pointer = out_view.data_ptr()
    assert not out_view.is_contiguous()
    result = flag_gems.tensordot_out(a, b, [1], [0], out=out_view)
    assert result is out_view
    assert out_view.data_ptr() == out_pointer
    utils.gems_assert_close(out_view, ref_out, dtype, reduce_dim=4)

    # Wrong (non-empty, mismatched) size -> resized with a deprecation warning.
    ref_out = torch.empty(2, 5, dtype=ref_a.dtype, device=ref_a.device)
    with pytest.warns(UserWarning):
        torch.tensordot(ref_a, ref_b, dims=([1], [0]), out=ref_out)
    out_wrong = torch.empty(2, 5, dtype=dtype, device=flag_gems.device)
    with pytest.warns(UserWarning):
        result = flag_gems.tensordot_out(a, b, [1], [0], out=out_wrong)
    assert result is out_wrong
    assert tuple(out_wrong.shape) == tuple(ref_out.shape) == (3, 5)
    utils.gems_assert_close(out_wrong, ref_out, dtype, reduce_dim=4)

    # Wrong dtype -> error (matches ATen rejecting a dtype-mismatched out).
    wrong_dtype = torch.float16 if dtype != torch.float16 else torch.float32
    out_wrong_dtype = torch.empty(3, 5, dtype=wrong_dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="dtype"):
        flag_gems.tensordot_out(a, b, [1], [0], out=out_wrong_dtype)

    # Wrong device -> error (only meaningful when a second device exists).
    if torch.cuda.is_available():
        out_cpu = torch.empty(3, 5, dtype=dtype, device="cpu")
        with pytest.raises(RuntimeError, match="device"):
            flag_gems.tensordot_out(a, b, [1], [0], out=out_cpu)
