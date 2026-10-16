# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

LSTM, GRU, RNN_RELU, RNN_TANH = 2, 3, 0, 1
CASES = [
    (16, 4, 128, 128, 1, False, LSTM, False),
    (16, 4, 128, 128, 2, False, LSTM, False),
    (16, 4, 128, 128, 1, True, LSTM, False),
    (16, 4, 128, 128, 1, False, LSTM, True),
    (16, 4, 128, 128, 1, False, GRU, False),
    (16, 4, 128, 128, 1, False, RNN_RELU, False),
    (16, 4, 128, 128, 1, False, RNN_TANH, False),
]


class CudnnRnnBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CASES)
        self.shape_desc = "seq,batch,input,hidden,layers,bidirectional,mode,batch_first"


def _case_fn(shape, dtype):
    del dtype
    config = tuple(shape)
    seq, batch, inp, hidden, layers, bidirectional, mode, batch_first = config
    yield base.BenchmarkCasePlan(
        shape={"input": [batch, seq, inp] if batch_first else [seq, batch, inp],
               "hx": [layers * (2 if bidirectional else 1), batch, hidden]},
        params={"mode": mode, "layers": layers, "bidirectional": bidirectional,
                "batch_first": batch_first},
        builder_args=(config,),
    )


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(0)
    seq, batch, inp_size, hidden, layers, bidirectional, mode, batch_first = plan.builder_args[0]
    kwargs = dict(num_layers=layers, bidirectional=bidirectional, batch_first=batch_first)
    if mode == LSTM:
        module = torch.nn.LSTM(inp_size, hidden, **kwargs)
    elif mode == GRU:
        module = torch.nn.GRU(inp_size, hidden, **kwargs)
    else:
        module = torch.nn.RNN(inp_size, hidden, nonlinearity="relu" if mode == RNN_RELU else "tanh", **kwargs)
    module = module.to(device=device, dtype=dtype)
    module.flatten_parameters()
    weights = tuple(module._flat_weights)
    directions = 2 if bidirectional else 1
    input_shape = (batch, seq, inp_size) if batch_first else (seq, batch, inp_size)
    value = torch.randn(input_shape, dtype=dtype, device=device)
    hx = torch.randn(layers * directions, batch, hidden, dtype=dtype, device=device)
    cx = torch.randn_like(hx) if mode == LSTM else None
    return (value, weights, 4, None, hx, cx, mode, hidden, 0, layers,
            batch_first, 0.0, False, bidirectional, [], None)


@pytest.mark.cudnn_rnn
def test_cudnn_rnn():
    CudnnRnnBenchmark(
        op_name="cudnn_rnn",
        torch_op=torch.ops.aten._cudnn_rnn,
        gems_op=flag_gems.cudnn_rnn,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
