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
from .generated_operator_utils import OperatorBenchmark


_SCALARS = (0.5, -1.25, 2.0)


def _tensor_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "other": list(dims)},
        params={"overload": "Tensor"},
        builder_args=(dims,),
    )


def _tensor_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = utils.generate_tensor_input(shape, dtype, device)
    other = utils.generate_tensor_input(shape, dtype, device)
    return inp, other


def _scalar_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for scalar in _SCALARS:
        yield base.BenchmarkCasePlan(
            shape={"self": list(dims)},
            params={"overload": "Scalar", "scalar": scalar},
            builder_args=(dims, scalar),
        )


def _scalar_build_inputs_fn(plan, dtype, device):
    shape, scalar = plan.builder_args
    torch.manual_seed(0)
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, scalar


@pytest.mark.multiply_
def test_multiply_tensor_():
    bench = OperatorBenchmark(
        op_name="multiply_",
        torch_op=lambda a, b: a.multiply_(b),
        case_fn=_tensor_case_fn,
        build_inputs_fn=_tensor_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()


@pytest.mark.multiply_
def test_multiply_scalar_():
    bench = OperatorBenchmark(
        op_name="multiply_",
        torch_op=lambda a, b: a.multiply_(b),
        case_fn=_scalar_case_fn,
        build_inputs_fn=_scalar_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
