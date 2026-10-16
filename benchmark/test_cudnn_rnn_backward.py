# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(3, 2, 4, 5), (5, 4, 8, 8), (8, 16, 64, 64)]


class CudnnRnnBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "seq,batch,input,hidden"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    seq, batch, inp, hidden = dims
    yield base.BenchmarkCasePlan(
        shape={"input": [seq, batch, inp], "hx": [1, batch, hidden]},
        params={"mode": 2, "output_mask": [True, True, True, True]},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(0)
    seq, batch, inp_size, hidden = plan.builder_args[0]
    value = torch.randn(seq, batch, inp_size, dtype=dtype, device=device)
    hx = torch.randn(1, batch, hidden, dtype=dtype, device=device)
    cx = torch.randn_like(hx)
    weights = [
        torch.randn(4 * hidden, inp_size, dtype=dtype, device=device),
        torch.randn(4 * hidden, hidden, dtype=dtype, device=device),
        torch.randn(4 * hidden, dtype=dtype, device=device),
        torch.randn(4 * hidden, dtype=dtype, device=device),
    ]
    output, hy, cy, reserve, weight_buf = torch.ops.aten._cudnn_rnn(
        value, weights, 4, None, hx, cx, 2, hidden, 0, 1, False,
        0.0, True, False, [], None,
    )
    return (value, weights, 4, weight_buf, hx, cx, output,
            torch.randn_like(output), torch.randn_like(hy), torch.randn_like(cy),
            2, hidden, 0, 1, False, 0.0, True, False, [], None, reserve,
            [True, True, True, True])


@pytest.mark.cudnn_rnn_backward
def test_cudnn_rnn_backward():
    bench = CudnnRnnBackwardBenchmark(
        op_name="cudnn_rnn_backward",
        torch_op=torch.ops.aten._cudnn_rnn_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.cudnn_rnn_backward)
    bench.run()
