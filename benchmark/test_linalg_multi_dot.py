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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

CORE_CHAINS = [
    (64, 256, 128),
    (256, 64, 512),
    (128, 512, 64, 256),
    (512, 64, 256, 128),
    (128, 256, 64, 512, 128),
    (256, 128, 512, 64, 256),
]
COMPREHENSIVE_CHAINS = [
    (1024, 64, 1024),
    (2048, 32, 2048, 32),
    (512, 2048, 64, 1024),
    (1024, 128, 2048, 64, 512),
    (256, 1024, 64, 2048, 128, 512),
    (512, 128, 1024, 64, 2048, 32, 512),
]


class LinalgMultiDotBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_CHAINS)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(COMPREHENSIVE_CHAINS)
        self.shape_desc = "matrix-chain dimensions (d0, d1, ..., dn)"


def _case_fn(dimensions, dtype):
    del dtype
    dims = tuple(dimensions)
    yield base.BenchmarkCasePlan(
        shape={"tensors": [[dims[i], dims[i + 1]] for i in range(len(dims) - 1)]},
        params={"chain_length": len(dims) - 1},
        builder_args=(dims,),
    )


def _build_tensors(dims, dtype, device):
    torch.manual_seed(0)
    return [
        torch.randn((dims[i], dims[i + 1]), dtype=dtype, device=device)
        for i in range(len(dims) - 1)
    ]


def _build_inputs_fn(plan, dtype, device):
    (dims,) = plan.builder_args
    return (_build_tensors(dims, dtype, device),)


@pytest.mark.linalg_multi_dot
def test_linalg_multi_dot():
    LinalgMultiDotBenchmark(
        op_name="linalg_multi_dot",
        torch_op=torch.ops.aten.linalg_multi_dot.default,
        gems_op=flag_gems.linalg_multi_dot,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
