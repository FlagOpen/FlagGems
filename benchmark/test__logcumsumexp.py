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


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    dim = 1
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims)},
        params={"dim": dim},
        builder_args=(dims, dim),
    )


def _out_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    dim = 1
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "out": list(dims)},
        params={"dim": dim},
        builder_args=(dims, dim),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, dim = plan.builder_args
    torch.manual_seed(0)
    return torch.randn(shape, dtype=dtype, device=device), dim


def _out_build_inputs_fn(plan, dtype, device):
    shape, dim = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    out = torch.empty(shape, dtype=dtype, device=device)
    return inp, dim, {"out": out}


class LogCumsumExpBenchmark(base.GenericBenchmark2DOnly, OperatorBenchmark):
    """Preserve the legacy 2-D shape policy with case-listing inputs."""

    def supports_cases(self):
        return True

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


@pytest.mark.underscore_logcumsumexp
def test__logcumsumexp():
    bench = LogCumsumExpBenchmark(
        op_name="_logcumsumexp",
        torch_op=torch._logcumsumexp,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.underscore_logcumsumexp_out
def test__logcumsumexp_out():
    bench = LogCumsumExpBenchmark(
        op_name="_logcumsumexp_out",
        torch_op=torch.ops.aten._logcumsumexp.out,
        case_fn=_out_case_fn,
        build_inputs_fn=_out_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
