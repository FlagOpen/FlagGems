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

REDUCE_MODES = ["sum", "prod", "mean", "amax", "amin"]


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


def _make_inputs(shape, dim, dtype, reduce):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if reduce == "prod":
        inp = inp.abs() + 0.5
        src = src.abs() + 0.5
    index = torch.randint(
        0, shape[dim], shape, dtype=torch.long, device=flag_gems.device
    )
    return inp, index, src


def _reference_inputs(inp, index, src):
    return (
        utils.to_reference(inp, True),
        utils.to_reference(index, False),
        utils.to_reference(src, True),
    )


@pytest.mark.scatter_reduce_
@pytest.mark.parametrize("shape,dim", [((4, 8), 1), ((2, 3, 5), -1)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduce", REDUCE_MODES)
@pytest.mark.parametrize("include_self", [True, False])
def test_scatter_reduce_inplace_contract(shape, dim, dtype, reduce, include_self):
    inp, index, src = _make_inputs(shape, dim, dtype, reduce)
    ref_inp, ref_index, ref_src = _reference_inputs(inp.clone(), index, src)
    index_before = index.clone()
    src_before = src.clone()
    pointer = inp.data_ptr()
    stride = inp.stride()
    storage_offset = inp.storage_offset()

    reference = torch.ops.aten.scatter_reduce_.two(
        ref_inp, dim, ref_index, ref_src, reduce, include_self=include_self
    )
    result = flag_gems.scatter_reduce_(
        inp, dim, index, src, reduce, include_self=include_self
    )

    assert result is inp
    assert result.data_ptr() == pointer
    assert result.stride() == stride
    assert result.storage_offset() == storage_offset
    normalized_dim = dim % len(shape)
    reduce_dim = src.shape[normalized_dim] if reduce in ("sum", "prod", "mean") else 1
    utils.gems_assert_close(
        result, reference, dtype, reduce_dim=reduce_dim
    )
    utils.gems_assert_equal(index, index_before)
    utils.gems_assert_equal(src, src_before)


@pytest.mark.scatter_reduce_
@pytest.mark.parametrize("reduce", REDUCE_MODES)
def test_scatter_reduce_noncontiguous_and_empty(reduce):
    dtype = torch.float32
    if reduce == "prod":
        base = torch.rand((5, 3), dtype=dtype, device=flag_gems.device) + 0.5
        src_base = torch.rand((5, 3), dtype=dtype, device=flag_gems.device) + 0.5
    else:
        base = torch.randn((5, 3), dtype=dtype, device=flag_gems.device)
        src_base = torch.randn((5, 3), dtype=dtype, device=flag_gems.device)
    inp = base.t()
    src = src_base.t()
    assert not inp.is_contiguous()
    assert not src.is_contiguous()
    index = torch.randint(0, inp.shape[1], inp.shape, device=inp.device)
    ref_inp, ref_index, ref_src = _reference_inputs(inp.clone(), index, src)
    result = flag_gems.scatter_reduce_(inp, 1, index, src, reduce)
    reference = torch.ops.aten.scatter_reduce_.two(
        ref_inp, 1, ref_index, ref_src, reduce
    )
    utils.gems_assert_close(result, reference, dtype, reduce_dim=src.shape[1])

    empty = torch.empty((2, 0), dtype=dtype, device=flag_gems.device)
    empty_index = torch.empty((2, 0), dtype=torch.long, device=flag_gems.device)
    empty_src = torch.empty_like(empty)
    empty_pointer = empty.data_ptr()
    returned = flag_gems.scatter_reduce_(
        empty, 1, empty_index, empty_src, reduce, include_self=False
    )
    assert returned is empty
    assert returned.data_ptr() == empty_pointer
    assert returned.shape == (2, 0)


@pytest.mark.scatter_reduce_
@pytest.mark.parametrize(
    "dim,reduce,exception",
    [(0, "invalid", RuntimeError), (3, "sum", IndexError)],
)
def test_scatter_reduce_rejects_invalid_arguments(dim, reduce, exception):
    inp, index, src = _make_inputs((2, 3), 1, torch.float32, "sum")
    with pytest.raises(exception):
        flag_gems.scatter_reduce_(inp, dim, index, src, reduce)


@pytest.mark.scatter_reduce_
def test_scatter_reduce_leaf_autograd_is_rejected():
    inp = torch.randn((2, 3), device=flag_gems.device, requires_grad=True)
    index = torch.tensor([[0, 1, 2], [0, 1, 2]], device=inp.device)
    src = torch.randn_like(inp)

    with pytest.raises(RuntimeError, match="leaf Variable"):
        torch.ops.aten.scatter_reduce_.two(inp.clone().detach().requires_grad_(True), 1, index, src, "sum")
    with pytest.raises(RuntimeError, match="leaf Variable"):
        with flag_gems.use_gems():
            torch.ops.aten.scatter_reduce_.two(inp, 1, index, src, "sum")
