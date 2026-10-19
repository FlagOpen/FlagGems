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

SHAPES = [(4, 4), (16, 1), (33, 16), (64, 64), (100, 40)]
DTYPES = [torch.float32] + ([torch.float64] if utils.fp64_is_supported else [])


def _make_inputs(batch_shape, n, k, dtype):
    matrix = torch.randn(*batch_shape, n, n, dtype=dtype, device=flag_gems.device)
    matrix = matrix @ matrix.mT + torch.eye(n, dtype=dtype, device=flag_gems.device) * n
    rhs = torch.randn(*batch_shape, n, k, dtype=dtype, device=flag_gems.device)
    lu, pivots = torch.linalg.lu_factor(matrix)
    return lu, pivots, rhs


@pytest.mark.linalg_lu_solve
@pytest.mark.parametrize("batch_shape", [(), (4,)])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_linalg_lu_solve(batch_shape, shape, dtype):
    n, k = shape
    lu, pivots, rhs = _make_inputs(batch_shape, n, k, dtype)
    ref_lu = utils.to_reference(lu, True)
    ref_pivots = utils.to_reference(pivots)
    ref_rhs = utils.to_reference(rhs, True)
    reference = torch.linalg.lu_solve(ref_lu, ref_pivots, ref_rhs)
    with flag_gems.use_gems():
        result = torch.linalg.lu_solve(lu, pivots, rhs)
    assert result is not lu
    assert result is not rhs
    if result.numel():
        assert result.data_ptr() != lu.data_ptr()
        assert result.data_ptr() != rhs.data_ptr()
    utils.gems_assert_close(result, reference, dtype)
