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

from . import base
from .generated_operator_utils import OperatorBenchmark

CASES = [(1, 1, 0), (8, 16, 0), (64, 128, 32), (256, 256, 1024)]
MAXBIT = 30


class SobolEngineFastForwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CASES)
        self.shape_desc = "dimension, n, num_generated"


def _case_fn(shape, dtype):
    del dtype
    dimension, n, num_generated = shape
    yield base.BenchmarkCasePlan(
        shape={"quasi": [dimension], "sobolstate": [dimension, MAXBIT]},
        params={"n": n, "dimension": dimension, "num_generated": num_generated},
        builder_args=(dimension, n, num_generated),
    )


def _build_inputs_fn(plan, dtype, device):
    del dtype
    dimension, n, num_generated = plan.builder_args
    torch.manual_seed(0)
    quasi = torch.zeros((dimension,), dtype=torch.int64, device=device)
    sobolstate = torch.randint(
        0, 2**30, (dimension, MAXBIT), dtype=torch.int64, device=device
    )
    return quasi, n, sobolstate, dimension, num_generated


def _cpu_reference(quasi, n, sobolstate, dimension, num_generated):
    # aten::_sobol_engine_ff_ has a CPU-only native kernel. Moving a cloned
    # state to CPU avoids invoking the unsupported native operator on a device.
    cpu_quasi = quasi.detach().to("cpu").clone()
    cpu_state = sobolstate.detach().to("cpu")
    return torch._sobol_engine_ff_(
        cpu_quasi, n, cpu_state, dimension, num_generated
    )


@pytest.mark.sobol_engine_ff_
def test_sobol_engine_ff_():
    SobolEngineFastForwardBenchmark(
        op_name="sobol_engine_ff_",
        torch_op=_cpu_reference,
        gems_op=flag_gems.ops._sobol_engine_ff_,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.int64],
        is_inplace=True,
        fresh_inputs=True,
    ).run()
