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

import math

import pytest
import torch

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark


_MAX_INPUT_ELEMENTS = 2**27
_INPUT_COUNT = 3


def _output_shape(shape):
    if len(shape) <= 1:
        return (math.prod(shape) if shape else 1, _INPUT_COUNT)
    result = list(shape)
    result[1] *= _INPUT_COUNT
    return tuple(result)


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    if math.prod(dims) > _MAX_INPUT_ELEMENTS:
        return
    yield base.BenchmarkCasePlan(
        shape={
            "inputs": [list(dims)] * _INPUT_COUNT,
            "out": list(_output_shape(dims)),
        },
        params={"input_count": _INPUT_COUNT},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    tensors = [
        utils.generate_tensor_input(shape, dtype, device) for _ in range(_INPUT_COUNT)
    ]
    out = torch.empty(_output_shape(shape), dtype=dtype, device=device)
    return tensors, {"out": out}


@pytest.mark.column_stack_out
def test_column_stack_out():
    bench = OperatorBenchmark(
        op_name="column_stack_out",
        torch_op=torch.column_stack,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        fresh_inputs=True,
    )
    bench.run()
