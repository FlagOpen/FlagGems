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


_VALUE_RANGES = [(-1.0, 1.0), (0.0, 1.0), (-1.0, 0.0)]
_EQUAL_SHAPES = [*utils.POINTWISE_SHAPES, (257,), (2, 0, 3)]
_BROADCAST_CASES = [
    pytest.param((7, 13), (1, 13), id="leading-dimension"),
    pytest.param((2, 3, 5), (1, 3, 1), id="multiple-dimensions"),
    pytest.param((2, 3, 5), (), id="scalar-tensor"),
    pytest.param((2, 0, 3), (1, 0, 1), id="zero-size"),
]


def _make_input(shape, dtype, low, high, *, requires_grad=False):
    tensor = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
        low, high
    )
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def _assert_inplace_result(result, mutated_input, original_pointer, reference, dtype):
    assert result is mutated_input
    assert result.data_ptr() == original_pointer
    assert result.shape == reference.shape
    assert result.dtype == dtype
    utils.gems_assert_close(result, reference, dtype)


@pytest.mark.inplace
@pytest.mark.multiply_
@pytest.mark.parametrize("shape", _EQUAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_multiply_tensor_equal_shape_(shape, dtype):
    torch.manual_seed(0)
    inp = _make_input(shape, dtype, -1.0, 1.0)
    other = _make_input(shape, dtype, -1.0, 1.0)
    original_pointer = inp.data_ptr()

    ref_inp = utils.to_reference(inp.clone(), True)
    ref_other = utils.to_reference(other, True)
    ref_out = ref_inp.multiply_(ref_other)

    with flag_gems.use_gems():
        res_out = inp.multiply_(other)

    _assert_inplace_result(res_out, inp, original_pointer, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.multiply_
@pytest.mark.parametrize("self_shape,other_shape", _BROADCAST_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("value_range", _VALUE_RANGES)
def test_multiply_tensor_broadcast_(self_shape, other_shape, dtype, value_range):
    torch.manual_seed(0)
    low, high = value_range
    inp = _make_input(self_shape, dtype, low, high)
    other = _make_input(other_shape, dtype, low, high)
    original_pointer = inp.data_ptr()

    ref_inp = utils.to_reference(inp.clone(), True)
    ref_other = utils.to_reference(other, True)
    ref_out = ref_inp.multiply_(ref_other)

    with flag_gems.use_gems():
        res_out = inp.multiply_(other)

    _assert_inplace_result(res_out, inp, original_pointer, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.multiply_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_multiply_scalar_(shape, scalar, dtype):
    torch.manual_seed(0)
    inp = _make_input(shape, dtype, -1.0, 1.0)
    original_pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_out = ref_inp.multiply_(scalar)

    with flag_gems.use_gems():
        res_out = inp.multiply_(scalar)

    _assert_inplace_result(res_out, inp, original_pointer, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.multiply_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_multiply_noncontiguous_(dtype):
    torch.manual_seed(0)
    base = _make_input((4, 7, 9), dtype, -1.0, 1.0)
    inp = base.transpose(0, 2)
    other = _make_input(inp.shape, dtype, -1.0, 1.0)
    original_pointer = inp.data_ptr()
    original_stride = inp.stride()

    ref_inp = utils.to_reference(inp.clone(), True)
    ref_other = utils.to_reference(other, True)
    ref_out = ref_inp.multiply_(ref_other)

    with flag_gems.use_gems():
        res_out = inp.multiply_(other)

    assert res_out.stride() == original_stride
    _assert_inplace_result(res_out, inp, original_pointer, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.multiply_
def test_multiply_nan_inf_():
    inp = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, 1.0, -1.0],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    other = torch.tensor(
        [1.0, -1.0, 0.0, float("inf"), float("nan"), -float("inf")],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    original_pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_other = utils.to_reference(other, True)
    ref_out = ref_inp.multiply_(ref_other)

    with flag_gems.use_gems():
        res_out = inp.multiply_(other)

    assert res_out is inp
    assert res_out.data_ptr() == original_pointer
    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.multiply_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_multiply_backward_(dtype):
    torch.manual_seed(0)
    inp_leaf = _make_input((4, 7), dtype, -1.0, 1.0, requires_grad=True)
    other = _make_input((4, 7), dtype, -1.0, 1.0, requires_grad=True)
    inp = inp_leaf + 0

    ref_leaf = utils.to_reference(inp_leaf.detach().clone(), True).requires_grad_(True)
    ref_other = utils.to_reference(other.detach().clone(), True).requires_grad_(True)
    ref_inp = ref_leaf + 0
    ref_out = ref_inp.multiply_(ref_other)

    with flag_gems.use_gems():
        res_out = inp.multiply_(other)

    upstream = _make_input((4, 7), dtype, -1.0, 1.0)
    ref_upstream = utils.to_reference(upstream, True)
    ref_grads = torch.autograd.grad(
        ref_out, (ref_leaf, ref_other), grad_outputs=ref_upstream
    )
    res_grads = torch.autograd.grad(
        res_out, (inp_leaf, other), grad_outputs=upstream
    )

    assert res_out is inp
    utils.gems_assert_close(res_grads[0], ref_grads[0], dtype)
    utils.gems_assert_close(res_grads[1], ref_grads[1], dtype)


@pytest.mark.inplace
@pytest.mark.multiply_
def test_multiply_rejects_nonbroadcastable_shape_():
    inp = torch.ones((2, 3), device=flag_gems.device)
    other = torch.ones((4,), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            inp.multiply_(other)
