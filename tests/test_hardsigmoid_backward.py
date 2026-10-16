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


@pytest.mark.hardsigmoid_backward
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardsigmoid_backward(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp_before = inp.clone()
    grad_before = grad_output.clone()
    ref_inp = utils.to_reference(inp, True)
    ref_grad_output = utils.to_reference(grad_output, True)

    ref_out = torch.ops.aten.hardsigmoid_backward(ref_grad_output, ref_inp)
    res_out = flag_gems.hardsigmoid_backward(grad_output, inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(inp, inp_before)
    utils.gems_assert_equal(grad_output, grad_before)


@pytest.mark.hardsigmoid_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardsigmoid_backward_boundaries(dtype):
    epsilon = torch.finfo(dtype).eps * 8
    inp = torch.tensor(
        [
            -float("inf"),
            -3.0,
            -3.0 + epsilon,
            -1.0,
            0.0,
            1.0,
            3.0 - epsilon,
            3.0,
            float("inf"),
            float("nan"),
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    grad_output = torch.linspace(-1.0, 1.0, inp.numel(), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_grad_output = utils.to_reference(grad_output, True)

    ref_out = torch.ops.aten.hardsigmoid_backward(ref_grad_output, ref_inp)
    res_out = flag_gems.hardsigmoid_backward(grad_output, inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.hardsigmoid_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardsigmoid_backward_noncontiguous(dtype):
    torch.manual_seed(0)
    inp = torch.randn((4, 7, 9), dtype=dtype, device=flag_gems.device).transpose(0, 2)
    grad_output = torch.randn((4, 7, 9), dtype=dtype, device=flag_gems.device).transpose(0, 2)
    assert not inp.is_contiguous()
    assert not grad_output.is_contiguous()
    ref_inp = utils.to_reference(inp, True)
    ref_grad_output = utils.to_reference(grad_output, True)

    ref_out = torch.ops.aten.hardsigmoid_backward(ref_grad_output, ref_inp)
    res_out = flag_gems.hardsigmoid_backward(grad_output, inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.hardsigmoid_backward
def test_hardsigmoid_backward_rejects_nonbroadcastable_shape():
    inp = torch.ones((2, 3), device=flag_gems.device)
    grad_output = torch.ones((4,), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.hardsigmoid_backward(grad_output, inp)
