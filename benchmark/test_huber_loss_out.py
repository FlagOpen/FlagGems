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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark


class HuberLossOutBenchmark(
    base.GenericBenchmark2DOnly, OperatorBenchmark
):
    pass


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    reductions = (1,) if base.Config.bench_level != consts.BenchLevel.COMPREHENSIVE else (0, 1, 2)
    for reduction in reductions:
        out_shape = dims if reduction == 0 else ()
        yield base.BenchmarkCasePlan(
            shape={"self": list(dims), "target": list(dims), "out": list(out_shape)},
            params={"reduction": reduction, "delta": 1.0},
            builder_args=(dims, out_shape, reduction),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, out_shape, reduction = plan.builder_args
    torch.manual_seed(0)
    inp = utils.generate_tensor_input(shape, dtype, device)
    target = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty(out_shape, dtype=dtype, device=device)
    return inp, target, reduction, 1.0, {"out": out}


@pytest.mark.huber_loss_out
def test_huber_loss_out():
    bench = HuberLossOutBenchmark(
        op_name="huber_loss_out",
        torch_op=torch.ops.aten.huber_loss.out,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
