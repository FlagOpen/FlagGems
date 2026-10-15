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


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input1": list(shape), "input2": list(shape)},
        params={"margin": 0.0, "reduction": 1},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    inp2 = torch.randn(shape, dtype=dtype, device=device)
    target = (torch.randint(0, 2, (shape[0],), device=device).to(dtype) * 2) - 1
    # aten::cosine_embedding_loss(input1, input2, target, margin, reduction)
    return inp1, inp2, target, 0.0, 1


@pytest.mark.cosine_embedding_loss
def test_cosine_embedding_loss():
    # 2D-only: cosine_embedding_loss needs (N, D) inputs with a 1D (N,) target.
    bench = base.GenericBenchmark2DOnly(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="cosine_embedding_loss",
        torch_op=torch.ops.aten.cosine_embedding_loss,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
