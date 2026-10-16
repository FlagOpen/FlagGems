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


_SHAPES = [
    (1024,),
    (512, 512),
    (20, 320, 15),
    (16, 128, 64),
]


class SpecialErfinvInplaceBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_SHAPES)


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims)},
        params={"domain": "[-0.9, 0.9]"},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.empty(shape, dtype=dtype, device=device).uniform_(-0.9, 0.9)
    return (inp,)


def _torch_reference_inplace(inp):
    return inp.copy_(torch.special.erfinv(inp))


@pytest.mark.special_erfinv_
def test_special_erfinv_():
    bench = SpecialErfinvInplaceBenchmark(
        op_name="special_erfinv_",
        torch_op=_torch_reference_inplace,
        gems_op=flag_gems.special_erfinv_,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
