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

SHAPES = [(2,3,8,8),(4,8,128,256),(2,4,512,512)]
PADDINGS = [(0,0,0,0),(1,1,2,2),(3,0,0,3)]

def _case_fn(shape,dtype):
    del dtype
    dims=tuple(shape)
    for padding in PADDINGS:
        left,right,top,bottom=padding; out_shape=(*dims[:-2],dims[-2]+top+bottom,dims[-1]+left+right)
        yield base.BenchmarkCasePlan(shape={"input":list(dims),"out":list(out_shape)},params={"padding":list(padding)},builder_args=(dims,padding,out_shape))

def _build_inputs_fn(plan,dtype,device):
    shape,padding,out_shape=plan.builder_args; torch.manual_seed(0)
    return torch.randn(shape,dtype=dtype,device=device),padding,{"out":torch.empty(out_shape,dtype=dtype,device=device)}


class ScopedBenchmark(OperatorBenchmark):
    def set_shapes(self,shape_file_path=None):
        self.shapes=list(SHAPES)


@pytest.mark.replication_pad2d_out
def test_replication_pad2d_out():
    ScopedBenchmark(
        op_name="replication_pad2d_out",
        torch_op=torch.ops.aten.replication_pad2d.out,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    ).run()
