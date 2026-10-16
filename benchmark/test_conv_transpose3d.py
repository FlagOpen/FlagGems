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

CASES = [(4,16,8,8,8,32,3,1,0,1),(2,32,8,16,16,16,3,2,1,1),(2,16,12,12,12,32,2,1,0,1),(1,32,16,16,16,16,3,2,1,2),(2,24,10,10,10,24,3,1,1,3)]

class ConvTranspose3dBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CASES)
        self.shape_desc = "N, Cin, D, H, W, Cout, K, stride, padding, groups"

def _case_fn(shape, dtype):
    del dtype
    case = tuple(shape)
    yield base.BenchmarkCasePlan(shape={"case": list(case)}, params={}, builder_args=(case,))

def _build_inputs_fn(plan, dtype, device):
    n, cin, d, h, w, cout, kernel, stride, padding, groups = plan.builder_args[0]
    torch.manual_seed(0)
    inp = torch.randn((n, cin, d, h, w), dtype=dtype, device=device)
    weight = torch.randn((cin, cout // groups, kernel, kernel, kernel), dtype=dtype, device=device)
    return inp, weight, None, stride, padding, 0, groups, 1

@pytest.mark.conv_transpose3d
def test_conv_transpose3d():
    ConvTranspose3dBenchmark(op_name="conv_transpose3d", torch_op=torch.nn.functional.conv_transpose3d, case_fn=_case_fn, build_inputs_fn=_build_inputs_fn, dtypes=consts.FLOAT_DTYPES).run()
