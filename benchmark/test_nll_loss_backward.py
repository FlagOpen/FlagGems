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

from . import base, consts


class NllLossBackwardBenchmark(base.GenericBenchmark2DOnly):
    DEFAULT_SHAPES = [(64, 64), (4096, 4096)]


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={
            "grad_output": [],
            "self": list(dims),
            "target": [dims[0]],
            "total_weight": [],
        },
        params={
            "weighted": False,
            "reduction": 1,
            "ignore_index": -100,
        },
        builder_args=(dims, False, 1),
    )
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield base.BenchmarkCasePlan(
            shape={
                "grad_output": [dims[0]],
                "self": list(dims),
                "target": [dims[0]],
                "weight": [dims[1]],
                "total_weight": [],
            },
            params={
                "weighted": True,
                "reduction": 0,
                "ignore_index": 1,
            },
            builder_args=(dims, True, 0),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, weighted, reduction = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = torch.randint(0, shape[1], (shape[0],), device=device)
    ignore_index = 1 if weighted else -100
    if weighted:
        target[0] = ignore_index
        weight = torch.rand(shape[1], dtype=dtype, device=device) + 0.25
        valid_targets = target[target != ignore_index]
        total_weight = weight[valid_targets].sum()
    else:
        weight = None
        total_weight = torch.tensor(
            target.numel(), dtype=dtype, device=device
        )
    grad_shape = target.shape if reduction == 0 else ()
    grad_output = torch.randn(grad_shape, dtype=dtype, device=device)
    return (
        grad_output,
        inp,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight,
    )


@pytest.mark.nll_loss_backward
def test_nll_loss_backward():
    bench = NllLossBackwardBenchmark(
        op_name="nll_loss_backward",
        torch_op=torch.ops.aten.nll_loss_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
