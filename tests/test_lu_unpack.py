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


LU_SHAPES = [(3, 3), (3, 4), (4, 3), (8, 8), (64, 64), (256, 128)]
LU_BATCH_SHAPES = [(2, 8, 8), (2, 3, 16, 8)]
LU_DTYPES = [torch.float32, torch.float64]


def _factor(shape, dtype):
    torch.manual_seed(0)
    matrix = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_matrix = utils.to_reference(matrix, True)
    lu, pivots = torch.linalg.lu_factor(matrix)
    ref_lu, ref_pivots = torch.linalg.lu_factor(ref_matrix)
    return lu, pivots, ref_lu, ref_pivots


def _assert_outputs(actual, expected, dtype, reduce_dim):
    p, l, u = actual
    ref_p, ref_l, ref_u = expected
    assert p.shape == ref_p.shape and p.dtype == ref_p.dtype
    assert l.shape == ref_l.shape and l.dtype == ref_l.dtype
    assert u.shape == ref_u.shape and u.dtype == ref_u.dtype
    utils.gems_assert_close(p, ref_p, dtype)
    utils.gems_assert_close(l, ref_l, dtype, reduce_dim=reduce_dim)
    utils.gems_assert_close(u, ref_u, dtype, reduce_dim=reduce_dim)


@pytest.mark.lu_unpack
@pytest.mark.parametrize("shape", LU_SHAPES + LU_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", LU_DTYPES)
@pytest.mark.parametrize(
    "unpack_data,unpack_pivots", [(True, True), (False, True), (True, False), (False, False)]
)
def test_lu_unpack(shape, dtype, unpack_data, unpack_pivots):
    lu, pivots, ref_lu, ref_pivots = _factor(shape, dtype)
    with flag_gems.use_gems():
        result = torch.ops.aten.lu_unpack(
            lu, pivots, unpack_data=unpack_data, unpack_pivots=unpack_pivots
        )
    reference = torch.ops.aten.lu_unpack(
        ref_lu,
        ref_pivots,
        unpack_data=unpack_data,
        unpack_pivots=unpack_pivots,
    )
    _assert_outputs(result, reference, dtype, min(shape[-2:]))

    p, l, u = result
    if not unpack_pivots:
        assert p.numel() == 0
    if not unpack_data:
        assert l.numel() == 0
        assert u.numel() == 0


@pytest.mark.lu_unpack
@pytest.mark.parametrize("dtype", LU_DTYPES)
def test_lu_unpack_does_not_mutate_inputs(dtype):
    lu, pivots, _, _ = _factor((8, 5), dtype)
    original_lu = lu.clone()
    original_pivots = pivots.clone()
    with flag_gems.use_gems():
        torch.ops.aten.lu_unpack(lu, pivots)
    utils.gems_assert_equal(lu, original_lu)
    utils.gems_assert_equal(pivots, original_pivots)


@pytest.mark.lu_unpack
def test_lu_unpack_rejects_invalid_pivots():
    torch.manual_seed(0)
    lu = torch.randn((4, 3), device=flag_gems.device)
    bad_pivots = torch.ones((2,), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.ops.aten.lu_unpack(lu, bad_pivots)
