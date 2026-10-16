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

# matrix_exp_backward operates on batches of square matrices, so we use
# square (..., n, n) shapes rather than the generic pointwise shapes.
MATRIX_EXP_SHAPES = [
    (2, 2),
    (5, 5),
    (8, 8),
    (16, 16),
    (4, 6, 6),
    (2, 3, 5, 5),
]


@pytest.mark.matrix_exp_backward
@pytest.mark.parametrize("shape", MATRIX_EXP_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matrix_exp_backward(shape, dtype):
    torch.manual_seed(0)
    self_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_self = utils.to_reference(self_inp, upcast=True)
    ref_grad = utils.to_reference(grad_inp, upcast=True)

    ref_out = torch.ops.aten.matrix_exp_backward(ref_self, ref_grad)
    res_out = flag_gems.matrix_exp_backward(self_inp, grad_inp)

    # matrix_exp_backward chains many matmuls over the 2n x 2n block matrix, so
    # accumulation error scales with the contraction dimension (cf. bmm tests
    # using reduce_dim=K).
    n = shape[-1]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=2 * n)
