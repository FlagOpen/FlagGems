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

from .conftest import QUICK_MODE

# conv_transpose1d test shapes: (input_shape, weight_shape)
# input: (N, in_channels, L_in)
# weight: (in_channels, out_channels/groups, kernel_width)
if QUICK_MODE:
    SHAPE_CONV_TRANSPOSE1D = [
        ((2, 4, 8), (4, 8, 3)),
    ]
    FLOAT_DTYPES = [torch.float32]
    STRIDES = [1]
    PADDINGS = [1]
else:
    SHAPE_CONV_TRANSPOSE1D = [
        ((2, 4, 8), (4, 8, 3)),
        ((4, 8, 16), (8, 16, 3)),
        ((2, 16, 32), (16, 32, 5)),
    ]
    FLOAT_DTYPES = [torch.float32, torch.float16]
    STRIDES = [1, 2]
    PADDINGS = [0, 1]


@pytest.mark.conv_transpose1d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV_TRANSPOSE1D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv_transpose1d(shape, kernel, stride, padding, dtype, monkeypatch):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)
    ref_out = torch.nn.functional.conv_transpose1d(
        ref_inp, ref_weight, bias=None, stride=stride, padding=padding, dilation=1
    )

    res_out = flag_gems.conv_transpose1d(
        inp, weight, bias=None, stride=stride, padding=padding, dilation=1
    )
    in_channels = kernel[0]
    out_channels = kernel[1]
    kernel_width = kernel[2]
    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=kernel_width * max(in_channels, out_channels),
    )


@pytest.mark.conv_transpose1d
@pytest.mark.parametrize("shape, kernel", SHAPE_CONV_TRANSPOSE1D)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv_transpose1d_bias(shape, kernel, stride, padding, dtype, monkeypatch):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)
    out_channels = kernel[1]
    kernel_width = kernel[2]
    bias = torch.randn(out_channels, dtype=dtype, device=flag_gems.device)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.conv_transpose1d(
        ref_inp, ref_weight, bias=ref_bias, stride=stride, padding=padding, dilation=1
    )

    res_out = flag_gems.conv_transpose1d(
        inp, weight, bias=bias, stride=stride, padding=padding, dilation=1
    )
    in_channels = kernel[0]
    out_channels = kernel[1]
    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=kernel_width * max(in_channels, out_channels),
    )


@pytest.mark.conv_transpose1d
@pytest.mark.parametrize(
    "shape, kernel, groups",
    [
        ((2, 8, 16), (8, 4, 3), 2),
        ((4, 12, 32), (12, 4, 3), 3),
    ],
)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv_transpose1d_groups(
    shape, kernel, groups, stride, padding, dtype, monkeypatch
):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.conv_transpose1d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=groups,
    )

    res_out = flag_gems.conv_transpose1d(
        inp,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=groups,
    )
    in_channels_per_group = kernel[0] // groups
    kernel_width = kernel[2]
    utils.gems_assert_close(
        res_out, ref_out, dtype, reduce_dim=kernel_width * in_channels_per_group
    )


@pytest.mark.conv_transpose1d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conv_transpose1d_output_padding_dilation_and_noncontiguous(dtype):
    torch.manual_seed(0)
    base = torch.randn((2, 8, 13), dtype=dtype, device=flag_gems.device)
    inp = base[:, ::2, :]
    weight = torch.randn((4, 6, 3), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((6,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref = torch.nn.functional.conv_transpose1d(
        ref_inp, ref_weight, ref_bias, stride=2, padding=2,
        output_padding=1, dilation=2, groups=1
    )
    result = flag_gems.conv_transpose1d(
        inp, weight, bias, stride=2, padding=2, output_padding=1,
        groups=1, dilation=2
    )
    utils.gems_assert_close(result, ref, dtype, reduce_dim=12)
    assert result.shape == ref.shape
    assert result.dtype == dtype


@pytest.mark.conv_transpose1d
def test_conv_transpose1d_invalid_groups():
    torch.manual_seed(0)
    inp = torch.randn((1, 4, 8), device=flag_gems.device)
    weight = torch.randn((4, 3, 3), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.conv_transpose1d(inp, weight, groups=3)
