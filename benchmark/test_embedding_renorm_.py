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


SHAPES = [(1024, 128), (4096, 256), (16384, 256), (32768, 512), (50000, 256), (50000, 512)]


class EmbeddingRenormBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"self": list(shape), "indices": [shape[0] // 2]},
        params={"max_norm": 1.0, "norm_type": 2.0},
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(0)
    (shape,) = plan.builder_args
    weight = torch.randn(shape, dtype=dtype, device=device) * 1.5
    indices = torch.randint(0, shape[0], (shape[0] // 2,), device=device)
    return weight, indices, 1.0, 2.0


@pytest.mark.embedding_renorm_
def test_embedding_renorm_():
    bench = EmbeddingRenormBenchmark(
        op_name="embedding_renorm_",
        torch_op=torch.ops.aten.embedding_renorm_,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
