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
from . import conftest as cfg


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


def _make_inputs(seq, batch, input_size, hidden_size, dtype, batch_first):
    shape = (batch, seq, input_size) if batch_first else (seq, batch, input_size)
    value = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    hx = torch.randn(1, batch, hidden_size, dtype=dtype, device=flag_gems.device)
    module = torch.nn.RNN(input_size, hidden_size, 1, nonlinearity="relu").to(
        dtype=dtype, device=flag_gems.device
    )
    return value, hx, tuple(module._flat_weights)


@pytest.mark.rnn_relu
@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="The current Triton rnn_relu kernel is CUDA-only",
)
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize(
    "seq,batch,input_size,hidden_size", [(4, 2, 8, 8), (8, 4, 16, 16), (3, 1, 16, 128)]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rnn_relu_values_and_metadata(
    seq, batch, input_size, hidden_size, dtype, batch_first
):
    value, hx, params = _make_inputs(
        seq, batch, input_size, hidden_size, dtype, batch_first
    )
    value_before = value.clone()
    hx_before = hx.clone()
    params_before = tuple(parameter.clone() for parameter in params)
    ref_value = utils.to_reference(value, True)
    ref_hx = utils.to_reference(hx, True)
    ref_params = tuple(utils.to_reference(parameter, True) for parameter in params)

    reference = torch.ops.aten.rnn_relu.input(
        ref_value, ref_hx, ref_params, True, 1, 0.0, False, False, batch_first
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.rnn_relu.input(
            value, hx, params, True, 1, 0.0, False, False, batch_first
        )

    expected_shape = (
        (batch, seq, hidden_size) if batch_first else (seq, batch, hidden_size)
    )
    assert len(result) == 2
    assert result[0].shape == expected_shape
    assert result[1].shape == (1, batch, hidden_size)
    assert result[0].dtype == dtype
    assert result[1].dtype == dtype
    input_storage_pointers = {
        value.untyped_storage().data_ptr(),
        hx.untyped_storage().data_ptr(),
        *(parameter.untyped_storage().data_ptr() for parameter in params),
    }
    output_storage_pointers = {
        result[0].untyped_storage().data_ptr(),
        result[1].untyped_storage().data_ptr(),
    }
    assert len(output_storage_pointers) == 2
    assert output_storage_pointers.isdisjoint(input_storage_pointers)
    utils.gems_assert_close(result[0], reference[0], dtype)
    utils.gems_assert_close(result[1], reference[1], dtype)
    utils.gems_assert_equal(value, value_before)
    utils.gems_assert_equal(hx, hx_before)
    for parameter, before in zip(params, params_before):
        utils.gems_assert_equal(parameter, before)


@pytest.mark.rnn_relu
@pytest.mark.skipif(
    cfg.TO_CPU or flag_gems.device != "cuda" or not torch.cuda.is_available(),
    reason="The current Triton rnn_relu kernel is CUDA-only",
)
def test_rnn_relu_autograd():
    from flag_gems.ops.rnn_relu import _params_unpack
    from flag_gems.ops.rnn_relu import rnn_relu as gems_rnn_relu

    seq, batch, input_size, hidden_size = 4, 2, 8, 8
    value, hx, params = _make_inputs(
        seq, batch, input_size, hidden_size, torch.float32, False
    )
    value_gems = value.detach().clone().requires_grad_(True)
    hx_gems = hx.detach().clone().requires_grad_(True)
    params_gems = tuple(
        parameter.detach().clone().requires_grad_(True) for parameter in params
    )
    output_gems, hidden_gems = gems_rnn_relu(
        value_gems, hx_gems, params_gems, True, 1, 0.0, True, False, False
    )
    gradients_gems = torch.autograd.grad(
        output_gems.sum() + hidden_gems.sum(),
        (value_gems, hx_gems, *params_gems),
    )

    value_ref = value.detach().clone().requires_grad_(True)
    hx_ref = hx.detach().clone().requires_grad_(True)
    params_ref = tuple(
        parameter.detach().clone().requires_grad_(True) for parameter in params
    )
    weight_ih, weight_hh, bias_ih, bias_hh = _params_unpack(params_ref, True)
    hidden = hx_ref[0].clone()
    outputs = []
    for index in range(seq):
        pre_activation = torch.addmm(bias_ih, value_ref[index], weight_ih.t())
        pre_activation = pre_activation + torch.addmm(
            bias_hh, hidden, weight_hh.t()
        )
        hidden = torch.relu(pre_activation)
        outputs.append(hidden)
    output_ref = torch.stack(outputs, dim=0)
    hidden_ref = hidden.unsqueeze(0)
    gradients_ref = torch.autograd.grad(
        output_ref.sum() + hidden_ref.sum(),
        (value_ref, hx_ref, *params_ref),
    )

    for actual, expected in zip(gradients_gems, gradients_ref):
        utils.gems_assert_close(actual, expected, torch.float32)


@pytest.mark.rnn_relu
@pytest.mark.parametrize(
    "override,exception,message",
    [
        ({"hx": None}, ValueError, "hx must be provided"),
        ({"params": None}, ValueError, "params must be provided"),
        ({"num_layers": 2}, NotImplementedError, "single-layer"),
        ({"dropout": 0.5}, NotImplementedError, "without dropout"),
        ({"bidirectional": True}, NotImplementedError, "unidirectional"),
    ],
)
def test_rnn_relu_rejects_unsupported_contracts(override, exception, message):
    from flag_gems.ops.rnn_relu import rnn_relu as gems_rnn_relu

    value, hx, params = _make_inputs(2, 1, 4, 4, torch.float32, False)
    arguments = {
        "hx": hx,
        "params": params,
        "has_biases": True,
        "num_layers": 1,
        "dropout": 0.0,
        "train": False,
        "bidirectional": False,
        "batch_first": False,
    }
    arguments.update(override)

    with pytest.raises(exception, match=message):
        gems_rnn_relu(value, **arguments)
