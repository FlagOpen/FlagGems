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
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "other": list(dims)},
        params={"same_size": True},
        builder_args=(dims, dims),
    )
    other = (*dims[:-1], dims[-1] + 1) if dims else (1,)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "other": list(other)},
        params={"same_size": False},
        builder_args=(dims, other),
    )


def _logical_shape_view(shape, dtype, device):
    storage = torch.empty((1,), dtype=dtype, device=device)
    return torch.as_strided(storage, size=shape, stride=(0,) * len(shape))


def _build_inputs_fn(plan, dtype, device):
    self_shape, other_shape = plan.builder_args
    return (
        _logical_shape_view(self_shape, dtype, device),
        _logical_shape_view(other_shape, dtype, device),
    )


@pytest.mark.is_same_size
def test_is_same_size():
    bench = OperatorBenchmark(
        op_name="is_same_size",
        torch_op=torch.ops.aten.is_same_size,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
