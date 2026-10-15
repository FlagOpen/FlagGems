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


def huber_loss_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape), "target": list(shape)},
        builder_args=(shape, None),
    )

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        for reduction in ("mean", "sum", "none"):
            yield base.BenchmarkCasePlan(
                shape={"input": list(shape), "target": list(shape)},
                params={"reduction": reduction},
                builder_args=(shape, reduction),
            )


def huber_loss_build_inputs_fn(plan, dtype, device):
    shape, reduction = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    target = utils.generate_tensor_input(shape, dtype, device)
    if reduction is None:
        return inp, target
    return inp, target, {"reduction": reduction}


@pytest.mark.huber_loss
def test_huber_loss():
    bench = base.GenericBenchmark2DOnly(
        op_name="huber_loss",
        case_fn=huber_loss_case_fn,
        build_inputs_fn=huber_loss_build_inputs_fn,
        torch_op=torch.nn.functional.huber_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
