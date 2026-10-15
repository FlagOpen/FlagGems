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

from . import base, consts, utils


def _case_fn(shape, cur_dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        builder_args=(shape,),
    )


def _take_along_dim_build_inputs_fn(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    dim = inp.ndim - 1
    index = torch.randint(0, inp.shape[dim], inp.shape, device=device)
    return inp, index, dim


def _take_along_dim_out_build_inputs_fn(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    dim = inp.ndim - 1
    index = torch.randint(0, inp.shape[dim], inp.shape, device=device)
    out = torch.empty(index.shape, dtype=cur_dtype, device=device)
    return inp, index, dim, {"out": out}


@pytest.mark.take_along_dim
def test_take_along_dim():
    bench = base.GenericBenchmark(
        op_name="take_along_dim",
        case_fn=_case_fn,
        build_inputs_fn=_take_along_dim_build_inputs_fn,
        torch_op=torch.take_along_dim,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.take_along_dim_out
def test_take_along_dim_out():
    bench = base.GenericBenchmark(
        op_name="take_along_dim_out",
        case_fn=_case_fn,
        build_inputs_fn=_take_along_dim_out_build_inputs_fn,
        torch_op=torch.take_along_dim,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
