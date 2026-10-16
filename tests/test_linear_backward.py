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


LINEAR_SHAPES = [(4, 16), (8, 32), (16, 64), (32, 128)]
LINEAR_OUT_FEATURES = [32, 64, 128, 256]


def _make_inputs(batch_shape, in_features, out_features, dtype):
    torch.manual_seed(0)
    input_tensor = torch.randn(
        (*batch_shape, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (*batch_shape, out_features), dtype=dtype, device=flag_gems.device
    )
    return input_tensor, grad_output, weight


def _reference(input_tensor, grad_output, weight, output_mask):
    ref_input = utils.to_reference(input_tensor, True)
    ref_grad_output = utils.to_reference(grad_output, True)
    ref_weight = utils.to_reference(weight, True)
    flat_input = ref_input.reshape(-1, ref_input.shape[-1])
    flat_grad_output = ref_grad_output.reshape(-1, ref_grad_output.shape[-1])
    batch_shape = ref_input.shape[:-1]
    grad_input = (
        (flat_grad_output @ ref_weight).reshape(*batch_shape, ref_weight.shape[-1])
        if output_mask[0]
        else None
    )
    grad_weight = (
        flat_grad_output.transpose(-2, -1) @ flat_input if output_mask[1] else None
    )
    grad_bias = flat_grad_output.sum(dim=0) if output_mask[2] else None
    return grad_input, grad_weight, grad_bias


def _reduction_dims(input_tensor, weight):
    batch_count = input_tensor.numel() // input_tensor.shape[-1]
    return weight.shape[0], batch_count, batch_count


def _assert_outputs(result, reference, dtype, reduction_dims):
    assert len(result) == 3
    for actual, expected, reduce_dim in zip(result, reference, reduction_dims):
        if expected is None:
            assert actual is None
        else:
            assert actual.shape == expected.shape
            assert actual.dtype == dtype
            utils.gems_assert_close(actual, expected, dtype, reduce_dim=reduce_dim)


@pytest.mark.linear_backward
@pytest.mark.parametrize("batch,in_features", LINEAR_SHAPES)
@pytest.mark.parametrize("out_features", LINEAR_OUT_FEATURES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linear_backward(batch, in_features, out_features, dtype):
    input_tensor, grad_output, weight = _make_inputs(
        (batch,), in_features, out_features, dtype
    )
    snapshots = tuple(
        utils.to_reference(tensor.clone(), False)
        for tensor in (input_tensor, grad_output, weight)
    )
    output_mask = (True, True, True)

    reference = _reference(input_tensor, grad_output, weight, output_mask)
    with flag_gems.use_gems():
        result = torch.ops.aten.linear_backward(
            input_tensor, grad_output, weight, output_mask
        )

    _assert_outputs(result, reference, dtype, _reduction_dims(input_tensor, weight))
    utils.gems_assert_equal(input_tensor, snapshots[0])
    utils.gems_assert_equal(grad_output, snapshots[1])
    utils.gems_assert_equal(weight, snapshots[2])


@pytest.mark.linear_backward
@pytest.mark.parametrize(
    "output_mask",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, False, True),
        (False, False, False),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linear_backward_output_mask(output_mask, dtype):
    input_tensor, grad_output, weight = _make_inputs((5,), 16, 24, dtype)
    reference = _reference(input_tensor, grad_output, weight, output_mask)

    with flag_gems.use_gems():
        result = torch.ops.aten.linear_backward(
            input_tensor, grad_output, weight, output_mask
        )

    _assert_outputs(result, reference, dtype, _reduction_dims(input_tensor, weight))


@pytest.mark.linear_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linear_backward_preserves_batch_dimensions(dtype):
    input_tensor, grad_output, weight = _make_inputs((2, 3), 16, 24, dtype)
    output_mask = (True, True, True)
    reference = _reference(input_tensor, grad_output, weight, output_mask)

    with flag_gems.use_gems():
        result = torch.ops.aten.linear_backward(
            input_tensor, grad_output, weight, output_mask
        )

    _assert_outputs(result, reference, dtype, _reduction_dims(input_tensor, weight))
    assert result[0].shape == input_tensor.shape


@pytest.mark.linear_backward
def test_linear_backward_rejects_invalid_output_mask_length():
    input_tensor, grad_output, weight = _make_inputs((2,), 4, 6, torch.float32)
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems.linear_backward(input_tensor, grad_output, weight, (True, False))
