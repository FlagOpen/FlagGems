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
from .generated_operator_utils import OperatorBenchmark

CASES = [(32,64,128,64,3,1,0,1),(64,48,256,128,5,2,2,1),(16,24,512,96,7,1,3,1),(8,16,1024,32,3,2,1,2),(4,8,2048,16,5,1,2,1)]

class ConvTranspose1dBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CASES)
        self.shape_desc = "N, Cin, L, Cout, K, stride, padding, groups"

def _case_fn(shape, dtype):
    del dtype
    case = tuple(shape)
    yield base.BenchmarkCasePlan(shape={"case": list(case)}, params={}, builder_args=(case,))

def _build_inputs_fn(plan, dtype, device):
    n, cin, length, cout, kernel, stride, padding, groups = plan.builder_args[0]
    torch.manual_seed(0)
    inp = torch.randn((n, cin, length), dtype=dtype, device=device)
    weight = torch.randn((cin, cout // groups, kernel), dtype=dtype, device=device)
    return inp, weight, None, stride, padding, 0, groups, 1

@pytest.mark.conv_transpose1d
def test_conv_transpose1d():
    ConvTranspose1dBenchmark(op_name="conv_transpose1d", torch_op=torch.nn.functional.conv_transpose1d, case_fn=_case_fn, build_inputs_fn=_build_inputs_fn, dtypes=consts.FLOAT_DTYPES).run()
