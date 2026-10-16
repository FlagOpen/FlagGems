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


@pytest.mark.leaky_relu_
@pytest.mark.parametrize("shape", [*utils.POINTWISE_SHAPES, (2, 0, 3)])
@pytest.mark.parametrize("negative_slope", [-0.25, 0.0, 0.01, 1.0])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_(shape, negative_slope, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    original_pointer = inp.data_ptr()
    reference = torch.nn.functional.leaky_relu_(
        utils.to_reference(inp.clone(), True), negative_slope=negative_slope
    )

    result = flag_gems.leaky_relu_(inp, negative_slope=negative_slope)

    assert result is inp
    assert result.data_ptr() == original_pointer
    assert result.dtype == dtype
    utils.gems_assert_close(result, reference, dtype)


@pytest.mark.leaky_relu_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_leaky_relu_noncontiguous_(dtype):
    torch.manual_seed(0)
    inp = torch.randn((4, 7, 9), dtype=dtype, device=flag_gems.device).transpose(0, 2)
    original_pointer = inp.data_ptr()
    original_stride = inp.stride()
    reference = torch.nn.functional.leaky_relu_(
        utils.to_reference(inp.clone(), True), negative_slope=0.2
    )

    result = flag_gems.leaky_relu_(inp, negative_slope=0.2)

    assert result is inp
    assert result.data_ptr() == original_pointer
    assert result.stride() == original_stride
    utils.gems_assert_close(result, reference, dtype)


@pytest.mark.leaky_relu_
def test_leaky_relu_autograd_():
    torch.manual_seed(0)
    base = torch.randn((11, 7), device=flag_gems.device, requires_grad=True)
    inp = base * 1.0
    grad = torch.randn_like(inp)
    ref_base = utils.to_reference(base, True).requires_grad_(True)
    ref_inp = ref_base * 1.0
    ref_grad = utils.to_reference(grad, True)
    ref_out = torch.nn.functional.leaky_relu_(ref_inp, negative_slope=0.1)
    (ref_input_grad,) = torch.autograd.grad(ref_out, ref_base, ref_grad)

    result = flag_gems.leaky_relu_(inp, negative_slope=0.1)
    (input_grad,) = torch.autograd.grad(result, base, grad)

    utils.gems_assert_close(result, ref_out, result.dtype)
    utils.gems_assert_close(input_grad, ref_input_grad, input_grad.dtype)
