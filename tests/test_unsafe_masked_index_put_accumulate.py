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


def _flat_to_per_dim_indices(flat_indices, inp_shape):
    strides, prod = [], 1
    for size in reversed(inp_shape):
        strides.append(prod)
        prod *= size
    return tuple((flat_indices // stride) % size for stride, size in zip(reversed(strides), inp_shape))


@pytest.mark.unsafe_masked_index_put_accumulate
@pytest.mark.parametrize("shape", utils._UNSAFE_MASKED_INDEX_PUT_SHAPES)
@pytest.mark.parametrize("dtype", [d for d in utils.FLOAT_DTYPES if d != torch.bfloat16])
def test_unsafe_masked_index_put_accumulate(shape, dtype):
    torch.manual_seed(0)
    inp_shape, mask_shape, indices_shape, values_shape = shape
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randint(0, 2, mask_shape, dtype=torch.int32, device=flag_gems.device)
    flat_indices = torch.randint(0, max(inp.numel(), 1), indices_shape, device=flag_gems.device)
    values = torch.randn(values_shape, dtype=dtype, device=flag_gems.device)
    indices = _flat_to_per_dim_indices(flat_indices, inp_shape)
    snapshots = (inp.clone(), mask.clone(), tuple(x.clone() for x in indices), values.clone())
    ref_out = torch._unsafe_masked_index_put_accumulate(
        utils.to_reference(inp, True),
        utils.to_reference(mask, False),
        tuple(utils.to_reference(x, False) for x in indices),
        utils.to_reference(values, True),
    )
    with flag_gems.use_gems():
        result = torch._unsafe_masked_index_put_accumulate(inp, mask, indices, values)
    utils.gems_assert_close(result, ref_out, dtype, atol=1e-2 if dtype == torch.float16 else 5e-3)
    utils.gems_assert_equal(inp, snapshots[0])
    utils.gems_assert_equal(mask, snapshots[1])
    for index, snapshot in zip(indices, snapshots[2]):
        utils.gems_assert_equal(index, snapshot)
    utils.gems_assert_equal(values, snapshots[3])
