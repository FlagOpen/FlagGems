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

SHAPES = [(2,3),(4,8),(8,16),(1,32),(4,64),(8,128)]

def _case_fn(shape, dtype):
    del dtype
    batch,width=shape; padding=(1,2)
    yield base.BenchmarkCasePlan(shape={"input":[batch,width],"grad_output":[batch,width+3]},params={"padding":list(padding)},builder_args=(batch,width,padding))

def _build_inputs_fn(plan,dtype,device):
    batch,width,padding=plan.builder_args; torch.manual_seed(0)
    inp=torch.randn((batch,width),dtype=dtype,device=device)
    grad=torch.ones((batch,width+sum(padding)),dtype=dtype,device=device)
    return grad,inp,padding


class ScopedBenchmark(OperatorBenchmark):
    def set_shapes(self,shape_file_path=None):
        self.shapes=list(SHAPES)


@pytest.mark.replication_pad1d_backward
def test_replication_pad1d_backward():
    ScopedBenchmark(
        op_name="replication_pad1d_backward",
        torch_op=torch.ops.aten.replication_pad1d_backward.default,
        gems_op=flag_gems.replication_pad1d_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
