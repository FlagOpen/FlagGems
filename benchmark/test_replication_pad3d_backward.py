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

SHAPES = [(2,3,8,16,16,1,1,1,1,1,1),(2,8,16,32,32,2,1,1,2,3,1),(1,16,32,32,32,1,1,1,1,1,1)]

def _case_fn(shape,dtype):
    del dtype
    n,c,d,h,w,*padding=shape; padding=tuple(padding)
    left,right,top,bottom,front,back=padding
    grad_shape=(n,c,d+front+back,h+top+bottom,w+left+right)
    yield base.BenchmarkCasePlan(shape={"input":[n,c,d,h,w],"grad_output":list(grad_shape)},params={"padding":list(padding)},builder_args=((n,c,d,h,w),padding,grad_shape))

def _build_inputs_fn(plan,dtype,device):
    shape,padding,grad_shape=plan.builder_args; torch.manual_seed(0)
    inp=torch.randn(shape,dtype=dtype,device=device)
    grad=torch.randn(grad_shape,dtype=dtype,device=device)
    return grad,inp,padding


class ScopedBenchmark(OperatorBenchmark):
    def set_shapes(self,shape_file_path=None):
        self.shapes=list(SHAPES)


@pytest.mark.replication_pad3d_backward
def test_replication_pad3d_backward():
    ScopedBenchmark(
        op_name="replication_pad3d_backward",
        torch_op=torch.ops.aten.replication_pad3d_backward.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
