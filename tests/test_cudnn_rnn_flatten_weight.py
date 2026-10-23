# Copyright 2026, The FlagOS Contributors.
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
from .conftest import TO_CPU

# (input_size, hidden_size, num_layers, bidirectional, bias)
RECURRENT_CONFIGS = [
    (4, 3, 1, False, True),
    (4, 3, 2, False, True),
    (4, 3, 3, True, True),
    (4, 3, 2, True, True),
    (8, 5, 2, False, False),
    (8, 5, 2, True, False),
    (16, 8, 2, True, True),
]

# (module class, cudnn mode, proj_size, nonlinearity)
RNN_MODES = [
    (torch.nn.RNN, 0, 0, "relu"),
    (torch.nn.RNN, 1, 0, "tanh"),
    (torch.nn.LSTM, 2, 0, None),
    (torch.nn.GRU, 3, 0, None),
    (torch.nn.LSTM, 2, 2, None),
]


def _make_weights(dtype, cfg, mode_info):
    """Build the flat weight list of an RNN module, matching cuDNN's layout."""
    cls, _, proj_size, nonlinearity = mode_info
    input_size, hidden_size, num_layers, bidirectional, bias = cfg
    kwargs = {
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "bias": bias,
    }
    if proj_size:
        kwargs["proj_size"] = proj_size
    if nonlinearity:
        kwargs["nonlinearity"] = nonlinearity
    module = cls(input_size, hidden_size, **kwargs)
    with torch.no_grad():
        for name, param in module.named_parameters():
            param.copy_(torch.randn_like(param))
    weights = [
        w.detach().clone().to(device=flag_gems.device, dtype=dtype)
        for w in module._flat_weights
    ]
    stride0 = (4 if bias else 2) + (1 if proj_size else 0)
    return weights, stride0


def _op_args(cfg, mode_info):
    input_size, hidden_size, num_layers, bidirectional, _ = cfg
    _, mode, proj_size, _ = mode_info
    _, stride0 = _make_weights(torch.float32, cfg, mode_info)
    return (
        stride0,
        input_size,
        mode,
        hidden_size,
        proj_size,
        num_layers,
        True,
        bidirectional,
    )


def _reference(weights, args, dtype):
    """cuDNN reference; fall back to an fp32 pass when the dtype is unsupported.

    The flat weight buffer is a pure copy, so casting the fp32 result back to
    `dtype` is exact.
    """
    try:
        return torch.ops.aten._cudnn_rnn_flatten_weight(
            [utils.to_reference(w.clone()) for w in weights], *args
        )
    except RuntimeError:
        if dtype == torch.float32:
            raise
        ref = torch.ops.aten._cudnn_rnn_flatten_weight(
            [utils.to_reference(w.float().clone().contiguous()) for w in weights],
            *args,
        )
        return ref.to(dtype)


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("cfg", RECURRENT_CONFIGS)
@pytest.mark.parametrize("mode_info", RNN_MODES, ids=lambda m: m[0].__name__)
def test_cudnn_rnn_flatten_weight(dtype, cfg, mode_info):
    if TO_CPU:
        pytest.skip(
            "_cudnn_rnn_flatten_weight is CUDA-only, cannot run in quick-cpu mode"
        )
    weights, _ = _make_weights(dtype, cfg, mode_info)
    args = _op_args(cfg, mode_info)

    ref_out = _reference(weights, args, dtype)

    res_weights = [w.clone() for w in weights]
    res_out = flag_gems.cudnn_rnn_flatten_weight(res_weights, *args)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.cudnn_rnn_flatten_weight_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("cfg", RECURRENT_CONFIGS)
@pytest.mark.parametrize("mode_info", RNN_MODES, ids=lambda m: m[0].__name__)
def test_cudnn_rnn_flatten_weight_out(dtype, cfg, mode_info):
    if TO_CPU:
        pytest.skip(
            "_cudnn_rnn_flatten_weight is CUDA-only, cannot run in quick-cpu mode"
        )
    weights, _ = _make_weights(dtype, cfg, mode_info)
    args = _op_args(cfg, mode_info)

    ref_out = _reference(weights, args, dtype)

    res_weights = [w.clone() for w in weights]
    out = torch.empty_like(ref_out)
    res_out = flag_gems.cudnn_rnn_flatten_weight_out(
        res_weights, *args, out=out
    )

    assert res_out.data_ptr() == out.data_ptr()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.cudnn_rnn_flatten_weight
@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
def test_cudnn_rnn_flatten_weight_other_dtypes(dtype):
    if TO_CPU:
        pytest.skip(
            "_cudnn_rnn_flatten_weight is CUDA-only, cannot run in quick-cpu mode"
        )
    """The buffer is a bitwise copy, so any fixed-width dtype must work."""
    cfg = (4, 3, 2, True, True)
    mode_info = (torch.nn.LSTM, 2, 0, None)
    weights, _ = _make_weights(torch.float32, cfg, mode_info)
    weights = [w.to(dtype) for w in weights]
    args = _op_args(cfg, mode_info)

    res_weights = [w.clone() for w in weights]
    res_out = flag_gems.cudnn_rnn_flatten_weight(res_weights, *args)

    # cuDNN cannot handle fp64/int64 weights, so derive the copy order from an
    # fp32 probe of the same shapes filled with 0..N-1: the flattened probe is
    # then the index permutation into `cat(weights)`, which is dtype
    # independent because the flat buffer is a pure copy.
    numel = 0
    probe = []
    for w in weights:
        piece = torch.arange(w.numel(), dtype=torch.float32, device=w.device) + numel
        probe.append(piece.reshape(w.shape).contiguous())
        numel += w.numel()
    order = torch.ops.aten._cudnn_rnn_flatten_weight([w.clone() for w in probe], *args)
    ref_out = torch.cat([w.reshape(-1) for w in weights])[order.long()]

    assert res_out.dtype == dtype
    utils.gems_assert_equal(res_out, ref_out.to(dtype))


@pytest.mark.cudnn_rnn_flatten_weight
def test_cudnn_rnn_flatten_weight_errors():
    if TO_CPU:
        pytest.skip(
            "_cudnn_rnn_flatten_weight is CUDA-only, cannot run in quick-cpu mode"
        )
    weights, stride0 = _make_weights(
        torch.float32, (4, 3, 1, False, True), (torch.nn.LSTM, 2, 0, None)
    )
    with pytest.raises(RuntimeError):
        flag_gems.cudnn_rnn_flatten_weight([], 4, 4, 2, 3, 0, 1, True, False)
    # Truncated weight list.
    with pytest.raises(RuntimeError):
        flag_gems.cudnn_rnn_flatten_weight(
            weights[:2], stride0, 4, 2, 3, 0, 1, True, False
        )
