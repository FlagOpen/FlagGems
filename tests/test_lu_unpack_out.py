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


LU_SHAPES = [(3, 3), (3, 4), (4, 3), (8, 8), (2, 8, 5)]
LU_DTYPES = [torch.float32, torch.float64]


def _factor(shape, dtype):
    torch.manual_seed(0)
    matrix = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_matrix = utils.to_reference(matrix, True)
    lu, pivots = torch.linalg.lu_factor(matrix)
    ref_lu, ref_pivots = torch.linalg.lu_factor(ref_matrix)
    return lu, pivots, ref_lu, ref_pivots


def _empty_outputs(shape, dtype, device, *, zero_sized=False):
    if zero_sized:
        return [torch.empty(0, dtype=dtype, device=device) for _ in range(3)]
    batch = shape[:-2]
    m, n = shape[-2:]
    k = min(m, n)
    return [
        torch.empty((*batch, m, m), dtype=dtype, device=device),
        torch.empty((*batch, m, k), dtype=dtype, device=device),
        torch.empty((*batch, k, n), dtype=dtype, device=device),
    ]


def _call_out(lu, pivots, outputs, *, unpack_data=True, unpack_pivots=True):
    with flag_gems.use_gems():
        return torch.ops.aten.lu_unpack.out(
            lu,
            pivots,
            unpack_data,
            unpack_pivots,
            P=outputs[0],
            L=outputs[1],
            U=outputs[2],
        )


def _call_reference(lu, pivots, outputs, *, unpack_data=True, unpack_pivots=True):
    return torch.ops.aten.lu_unpack.out(
        lu,
        pivots,
        unpack_data,
        unpack_pivots,
        P=outputs[0],
        L=outputs[1],
        U=outputs[2],
    )


@pytest.mark.lu_unpack_out
@pytest.mark.parametrize("shape", LU_SHAPES)
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack_out_identity_and_values(shape, dtype):
    lu, pivots, ref_lu, ref_pivots = _factor(shape, dtype)
    outputs = _empty_outputs(shape, dtype, flag_gems.device)
    ref_outputs = _empty_outputs(shape, dtype, ref_lu.device)
    result = _call_out(lu, pivots, outputs)
    reference = _call_reference(ref_lu, ref_pivots, ref_outputs)

    assert all(value is out for value, out in zip(result, outputs))
    assert all(value is out for value, out in zip(reference, ref_outputs))
    k = min(shape[-2:])
    for value, expected in zip(result, reference):
        utils.gems_assert_close(value, expected, dtype, reduce_dim=k)


@pytest.mark.lu_unpack_out
@pytest.mark.parametrize("shape", [(4, 3), (2, 8, 5)])
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack_out_resizes_outputs(shape, dtype):
    lu, pivots, ref_lu, ref_pivots = _factor(shape, dtype)
    outputs = _empty_outputs(shape, dtype, flag_gems.device, zero_sized=True)
    ref_outputs = _empty_outputs(shape, dtype, ref_lu.device, zero_sized=True)
    result = _call_out(lu, pivots, outputs)
    reference = _call_reference(ref_lu, ref_pivots, ref_outputs)

    assert all(value is out for value, out in zip(result, outputs))
    for value, expected in zip(result, reference):
        assert value.shape == expected.shape
        utils.gems_assert_close(value, expected, dtype, reduce_dim=min(shape[-2:]))


@pytest.mark.lu_unpack_out
@pytest.mark.skipif(not utils.fp64_is_supported, reason="FP64 is not supported")
def test_lu_unpack_out_float_cast():
    shape = (8, 5)
    lu, pivots, ref_lu, ref_pivots = _factor(shape, torch.float64)
    outputs = _empty_outputs(shape, torch.float32, flag_gems.device)
    ref_outputs = _empty_outputs(shape, torch.float32, ref_lu.device)
    result = _call_out(lu, pivots, outputs)
    reference = _call_reference(ref_lu, ref_pivots, ref_outputs)

    for value, expected in zip(result, reference):
        utils.gems_assert_close(value, expected, torch.float32, reduce_dim=5)


@pytest.mark.lu_unpack_out
def test_lu_unpack_out_preserves_noncontiguous_storage():
    shape = (8, 5)
    lu, pivots, ref_lu, ref_pivots = _factor(shape, torch.float32)
    p = torch.empty((8, 8), device=flag_gems.device).T
    l = torch.empty((5, 8), device=flag_gems.device).T
    u = torch.empty((5, 5), device=flag_gems.device).T
    ref_p = torch.empty((8, 8), device=ref_lu.device).T
    ref_l = torch.empty((5, 8), device=ref_lu.device).T
    ref_u = torch.empty((5, 5), device=ref_lu.device).T
    outputs = [p, l, u]
    pointers = [value.data_ptr() for value in outputs]
    strides = [value.stride() for value in outputs]

    result = _call_out(lu, pivots, outputs)
    reference = _call_reference(ref_lu, ref_pivots, [ref_p, ref_l, ref_u])

    assert [value.data_ptr() for value in result] == pointers
    assert [value.stride() for value in result] == strides
    for value, expected in zip(result, reference):
        utils.gems_assert_close(value, expected, torch.float32, reduce_dim=5)


@pytest.mark.lu_unpack_out
def test_lu_unpack_out_rejects_integral_outputs():
    shape = (4, 3)
    lu, pivots, _, _ = _factor(shape, torch.float32)
    outputs = _empty_outputs(shape, torch.int32, flag_gems.device)
    with pytest.raises(RuntimeError):
        _call_out(lu, pivots, outputs)
