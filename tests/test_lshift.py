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


_SHAPES = [*utils.POINTWISE_SHAPES, (2, 0, 3)]
_BROADCAST_CASES = [
    pytest.param((7, 13), (1, 13), id="leading-dimension"),
    pytest.param((2, 3, 5), (1, 3, 1), id="multiple-dimensions"),
    pytest.param((2, 3, 5), (), id="scalar-tensor"),
    pytest.param((2, 0, 3), (1, 0, 1), id="zero-size"),
]


def _make_values(shape, dtype):
    return torch.randint(0, 0x00FF, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )


def _make_shifts(shape, dtype):
    return torch.randint(0, 8, shape, dtype=dtype, device="cpu").to(flag_gems.device)


@pytest.mark.lshift
@pytest.mark.lshift_tensor
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_lshift_tensor(shape, dtype):
    torch.manual_seed(0)
    inp = _make_values(shape, dtype)
    other = _make_shifts(shape, dtype)
    inp_before = inp.clone()
    other_before = other.clone()
    ref_inp = utils.to_reference(inp, False)
    ref_other = utils.to_reference(other, False)

    ref_out = torch.ops.aten.__lshift__.Tensor(ref_inp, ref_other)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.__lshift__.Tensor(inp, other)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inp, inp_before)
    utils.gems_assert_equal(other, other_before)


@pytest.mark.lshift
@pytest.mark.lshift_tensor
@pytest.mark.parametrize("self_shape,other_shape", _BROADCAST_CASES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_lshift_tensor_broadcast(self_shape, other_shape, dtype):
    torch.manual_seed(0)
    inp = _make_values(self_shape, dtype)
    other = _make_shifts(other_shape, dtype)
    ref_inp = utils.to_reference(inp, False)
    ref_other = utils.to_reference(other, False)

    ref_out = torch.ops.aten.__lshift__.Tensor(ref_inp, ref_other)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.__lshift__.Tensor(inp, other)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.lshift
@pytest.mark.lshift_scalar
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
@pytest.mark.parametrize("shift", [0, 1, 2, 7])
def test_lshift_scalar(dtype, shift):
    torch.manual_seed(0)
    inp = _make_values((11, 17), dtype)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.ops.aten.__lshift__.Scalar(ref_inp, shift)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.__lshift__.Scalar(inp, shift)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.lshift
@pytest.mark.lshift_tensor
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_lshift_noncontiguous(dtype):
    torch.manual_seed(0)
    inp = _make_values((4, 7, 9), dtype).transpose(0, 2)
    other = _make_shifts((4, 7, 9), dtype).transpose(0, 2)
    assert not inp.is_contiguous()
    assert not other.is_contiguous()
    ref_inp = utils.to_reference(inp, False)
    ref_other = utils.to_reference(other, False)

    ref_out = torch.ops.aten.__lshift__.Tensor(ref_inp, ref_other)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.__lshift__.Tensor(inp, other)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.lshift
@pytest.mark.lshift_tensor
def test_lshift_rejects_nonbroadcastable_shape():
    inp = torch.ones((2, 3), dtype=torch.int32, device=flag_gems.device)
    other = torch.ones((4,), dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.ops.aten.__lshift__.Tensor(inp, other)
