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

# (time, batch, in_channels, out_channels, kernel_width, pad)
CONV_TBC_SHAPES = [
    (7, 3, 4, 5, 3, 1),
    (10, 2, 6, 3, 4, 0),
    (5, 1, 3, 2, 5, 3),
    (9, 4, 2, 7, 2, 2),
    (32, 8, 16, 32, 3, 1),
    (64, 4, 32, 16, 5, 2),
    (128, 2, 8, 8, 1, 0),
]


@pytest.mark.conv_tbc_backward
@pytest.mark.parametrize("shape", CONV_TBC_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conv_tbc_backward(shape, dtype):
    torch.manual_seed(0)
    ilen, batch, in_c, out_c, kw, pad = shape
    olen = ilen - kw + 1 + 2 * pad

    inp = torch.randn((ilen, batch, in_c), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((kw, in_c, out_c), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((out_c,), dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(
        (olen, batch, out_c), dtype=dtype, device=flag_gems.device
    )

    ref_grad_output = utils.to_reference(grad_output, True)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_gi, ref_gw, ref_gb = torch.ops.aten.conv_tbc_backward(
        ref_grad_output, ref_inp, ref_weight, ref_bias, pad
    )
    res_gi, res_gw, res_gb = flag_gems.conv_tbc_backward(
        grad_output, inp, weight, bias, pad
    )

    # grad_input / grad_weight accumulate over reduction dims; scale tolerance.
    utils.gems_assert_close(res_gi, ref_gi, dtype, reduce_dim=kw * out_c)
    utils.gems_assert_close(res_gw, ref_gw, dtype, reduce_dim=olen * batch)
    utils.gems_assert_close(res_gb, ref_gb, dtype, reduce_dim=olen * batch)


@pytest.mark.conv_tbc_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conv_tbc_backward_noncontiguous_and_metadata(dtype):
    torch.manual_seed(0)
    inp_base = torch.randn((8, 2, 8), dtype=dtype, device=flag_gems.device)
    inp = inp_base[:, :, ::2]
    weight_base = torch.randn((3, 8, 5), dtype=dtype, device=flag_gems.device)
    weight = weight_base[:, ::2, :]
    bias = torch.randn((5,), dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn((8, 2, 5), dtype=dtype, device=flag_gems.device)
    ref = torch.ops.aten.conv_tbc_backward(
        utils.to_reference(grad_output, True), utils.to_reference(inp, True),
        utils.to_reference(weight, True), utils.to_reference(bias, True), 1
    )
    result = flag_gems.conv_tbc_backward(grad_output, inp, weight, bias, 1)
    for actual, expected in zip(result, ref):
        utils.gems_assert_close(actual, expected, dtype, reduce_dim=16)
        assert actual.shape == expected.shape
        assert actual.dtype == dtype


@pytest.mark.conv_tbc_backward
def test_conv_tbc_backward_invalid_grad_output_shape():
    torch.manual_seed(0)
    inp = torch.randn((7, 2, 4), device=flag_gems.device)
    weight = torch.randn((3, 4, 5), device=flag_gems.device)
    bias = torch.randn((5,), device=flag_gems.device)
    grad_output = torch.randn((6, 2, 4), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.conv_tbc_backward(grad_output, inp, weight, bias, 1)
