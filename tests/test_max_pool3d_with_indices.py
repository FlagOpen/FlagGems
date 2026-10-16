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

MAXPOOL3D_CONFIGS = [
    ((2, 3, 8, 9, 10), 3, 2, 1, 1, False),
    ((1, 2, 7, 8, 9), (2, 3, 2), (1, 2, 1), (0, 1, 0), 1, False),
    ((1, 1, 9, 9, 9), 2, 1, 0, 2, False),
    ((2, 4, 7, 7, 7), 3, 2, 1, 1, True),
]


def _reference(inp, kernel_size, stride, padding, dilation, ceil_mode):
    return torch.ops.aten.max_pool3d_with_indices(
        inp,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )


@pytest.mark.max_pool3d_with_indices
@pytest.mark.parametrize(
    "shape,kernel_size,stride,padding,dilation,ceil_mode", MAXPOOL3D_CONFIGS
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_max_pool3d_with_indices(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype
):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp_before = inp.clone()
    expected_output, expected_indices = _reference(
        utils.to_reference(inp, True),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )

    with flag_gems.use_gems():
        output, indices = torch.ops.aten.max_pool3d_with_indices(
            inp, kernel_size, stride, padding, dilation, ceil_mode
        )

    utils.gems_assert_close(output, expected_output, dtype)
    utils.gems_assert_equal(indices, expected_indices)
    utils.gems_assert_equal(inp, inp_before)
    assert output.shape == indices.shape == expected_output.shape
    assert output.dtype == dtype
    assert indices.dtype == torch.int64
    assert output is not inp
    assert indices is not inp
    input_storage_pointer = inp.untyped_storage().data_ptr()
    assert output.untyped_storage().data_ptr() != input_storage_pointer
    assert indices.untyped_storage().data_ptr() != input_storage_pointer


@pytest.mark.max_pool3d_with_indices
@pytest.mark.parametrize("shape", [(2, 3, 7, 9, 11), (3, 7, 9, 11)])
def test_max_pool3d_with_indices_default_stride(shape):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    expected = _reference(utils.to_reference(inp, True), 2, [], 0, 1, False)

    with flag_gems.use_gems():
        actual = torch.ops.aten.max_pool3d_with_indices(inp, 2, [], 0, 1, False)

    utils.gems_assert_close(actual[0], expected[0], torch.float32)
    utils.gems_assert_equal(actual[1], expected[1])


@pytest.mark.max_pool3d_with_indices
def test_max_pool3d_with_indices_noncontiguous_input():
    torch.manual_seed(0)
    storage = torch.randn((1, 2, 8, 9, 20), device=flag_gems.device)
    inp = storage[..., ::2]
    assert not inp.is_contiguous()
    expected = _reference(utils.to_reference(inp, True), 3, 2, 1, 1, False)

    with flag_gems.use_gems():
        actual = torch.ops.aten.max_pool3d_with_indices(inp, 3, 2, 1, 1, False)

    utils.gems_assert_close(actual[0], expected[0], torch.float32)
    utils.gems_assert_equal(actual[1], expected[1])


@pytest.mark.max_pool3d_with_indices
def test_max_pool3d_with_indices_autograd_matches_reference():
    torch.manual_seed(0)
    inp = torch.randn(
        (1, 2, 6, 6, 6),
        dtype=torch.float32,
        device=flag_gems.device,
        requires_grad=True,
    )
    ref_inp = utils.to_reference(inp, True).detach().requires_grad_(True)
    expected_output, expected_indices = torch.ops.aten.max_pool3d_with_indices(
        ref_inp, 2, 2, 0, 1, False
    )

    with flag_gems.use_gems():
        output, indices = torch.ops.aten.max_pool3d_with_indices(
            inp, 2, 2, 0, 1, False
        )

    torch.manual_seed(1)
    output_grad = torch.randn_like(output)
    expected_grad = torch.autograd.grad(
        expected_output,
        ref_inp,
        utils.to_reference(output_grad, True),
    )[0]
    actual_grad = torch.autograd.grad(output, inp, output_grad)[0]

    assert output.requires_grad
    assert not indices.requires_grad
    utils.gems_assert_equal(indices, expected_indices)
    utils.gems_assert_close(actual_grad, expected_grad, torch.float32)


@pytest.mark.max_pool3d_with_indices
@pytest.mark.parametrize(
    "kernel_size,stride,padding,dilation",
    [(0, 1, 0, 1), (2, 0, 0, 1), (2, 1, -1, 1), (2, 1, 0, 0)],
)
def test_max_pool3d_with_indices_invalid_parameters(
    kernel_size, stride, padding, dilation
):
    inp = torch.ones((1, 1, 5, 5, 5), device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises((ValueError, RuntimeError)):
        torch.ops.aten.max_pool3d_with_indices(
            inp, kernel_size, stride, padding, dilation, False
        )


@pytest.mark.max_pool3d_with_indices
def test_max_pool3d_with_indices_rejects_invalid_rank():
    inp = torch.ones((2, 5, 5), device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises((ValueError, RuntimeError)):
        torch.ops.aten.max_pool3d_with_indices(inp, 2, 2, 0, 1, False)
