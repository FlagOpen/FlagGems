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


MIXED_DTYPES_LINEAR_DTYPES = [
    dtype for dtype in utils.FLOAT_DTYPES if dtype != torch.float32
]
MIXED_DTYPES_LINEAR_SHAPES = [
    (16, 128, 64, "int8", True, "none"),
    (32, 256, 128, "int8", False, "relu"),
    (16, 128, 64, "int8", True, "silu"),
    (16, 128, 128, "int4", True, "none"),
    (32, 256, 256, "int4", True, "relu"),
    (64, 256, 128, "int8", True, "silu"),
]


def _ref_mixed_dtypes_linear(
    input, weight, scale, mode, bias, activation, output_dtype
):
    k_size = weight.shape[0]
    n_size = scale.shape[0]
    if mode == "int8":
        unpacked = weight.to(torch.int32) - 128
    else:
        low = (weight.to(torch.int32) & 0xF) - 8
        high = ((weight.to(torch.int32) >> 4) & 0xF) - 8
        unpacked = torch.stack([low, high], dim=-1).reshape(k_size, n_size)

    if output_dtype == torch.float16:
        accumulator = input @ unpacked.to(output_dtype).to(input.dtype)
        accumulator = accumulator * scale
    else:
        dequantized = (unpacked.to(scale.dtype) * scale.unsqueeze(0)).to(output_dtype)
        accumulator = input @ dequantized.to(input.dtype)

    if bias is not None:
        accumulator = accumulator + bias
    if activation == "relu":
        accumulator = torch.relu(accumulator)
    elif activation == "silu":
        accumulator = accumulator * torch.sigmoid(accumulator)
    return accumulator


def _make_inputs(shape, dtype):
    m_size, k_size, n_size, mode, has_bias, activation = shape
    torch.manual_seed(0)
    input_tensor = torch.randn(
        (m_size, k_size), dtype=dtype, device=flag_gems.device
    )
    ncols = n_size if mode == "int8" else n_size // 2
    weight = torch.randint(
        0, 256, (k_size, ncols), dtype=torch.uint8, device=flag_gems.device
    )
    scale = torch.randn((n_size,), dtype=dtype, device=flag_gems.device)
    bias = (
        torch.randn((n_size,), dtype=dtype, device=flag_gems.device)
        if has_bias
        else None
    )
    return input_tensor, weight, scale, bias, activation


@pytest.mark.mixed_dtypes_linear
@pytest.mark.parametrize("shape", MIXED_DTYPES_LINEAR_SHAPES)
@pytest.mark.parametrize("dtype", MIXED_DTYPES_LINEAR_DTYPES)
def test_mixed_dtypes_linear(shape, dtype):
    input_tensor, weight, scale, bias, activation = _make_inputs(shape, dtype)
    input_before = utils.to_reference(input_tensor.clone(), False)
    weight_before = utils.to_reference(weight.clone(), False)
    scale_before = utils.to_reference(scale.clone(), False)
    bias_before = utils.to_reference(bias.clone(), False) if bias is not None else None
    mode = shape[3]

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, False)
    ref_scale = utils.to_reference(scale, True)
    ref_bias = utils.to_reference(bias, True)
    ref_out = _ref_mixed_dtypes_linear(
        ref_input,
        ref_weight,
        ref_scale,
        mode,
        ref_bias,
        activation,
        dtype,
    )
    res_out = flag_gems.mixed_dtypes_linear(
        input_tensor, weight, scale, bias=bias, activation=activation
    )

    assert res_out.shape == (*input_tensor.shape[:-1], scale.shape[0])
    assert res_out.dtype == dtype
    assert res_out.data_ptr() != input_tensor.data_ptr()
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=input_tensor.shape[-1])
    utils.gems_assert_equal(input_tensor, input_before)
    utils.gems_assert_equal(weight, weight_before)
    utils.gems_assert_equal(scale, scale_before)
    if bias is not None:
        utils.gems_assert_equal(bias, bias_before)


@pytest.mark.mixed_dtypes_linear
@pytest.mark.parametrize("dtype", MIXED_DTYPES_LINEAR_DTYPES)
def test_mixed_dtypes_linear_preserves_batch_dimensions(dtype):
    torch.manual_seed(0)
    input_tensor = torch.randn((2, 3, 64), dtype=dtype, device=flag_gems.device)
    weight = torch.randint(
        0, 256, (64, 32), dtype=torch.uint8, device=flag_gems.device
    )
    scale = torch.randn((32,), dtype=dtype, device=flag_gems.device)
    ref_out = _ref_mixed_dtypes_linear(
        utils.to_reference(input_tensor, True),
        utils.to_reference(weight, False),
        utils.to_reference(scale, True),
        "int8",
        None,
        "none",
        dtype,
    )

    res_out = flag_gems.mixed_dtypes_linear(input_tensor, weight, scale)

    assert res_out.shape == (2, 3, 32)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=64)


@pytest.mark.mixed_dtypes_linear
@pytest.mark.parametrize(
    "invalid", ["input_dtype", "weight_dtype", "scale_dtype", "activation", "columns"]
)
def test_mixed_dtypes_linear_rejects_invalid_contract(invalid):
    torch.manual_seed(0)
    input_tensor = torch.randn(
        (4, 32), dtype=torch.float16, device=flag_gems.device
    )
    weight = torch.randint(
        0, 256, (32, 16), dtype=torch.uint8, device=flag_gems.device
    )
    scale = torch.randn((16,), dtype=torch.float16, device=flag_gems.device)
    activation = None
    if invalid == "input_dtype":
        input_tensor = input_tensor.float()
    elif invalid == "weight_dtype":
        weight = weight.to(torch.int16)
    elif invalid == "scale_dtype":
        scale = scale.float()
    elif invalid == "activation":
        activation = "gelu"
    else:
        weight = weight[:, :-1]

    with pytest.raises(AssertionError):
        flag_gems.mixed_dtypes_linear(
            input_tensor, weight, scale, activation=activation
        )
