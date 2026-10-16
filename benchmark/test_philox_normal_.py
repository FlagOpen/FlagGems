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


_SHAPES = [(1024,), (4096,), (16, 33), (1024, 1024)]
_MEAN_STD = ((0.0, 1.0), (-1.5, 0.5))
_HAS_NATIVE_PHILOX = hasattr(torch.ops.aten, "_philox_normal_")


class PhiloxNormalBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(_SHAPES)
        self.shape_desc = "shape"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for mean, std in _MEAN_STD:
        yield base.BenchmarkCasePlan(
            shape={"self": list(dims), "key": [2]},
            params={"mean": mean, "std": std},
            builder_args=(dims, mean, std),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, mean, std = plan.builder_args
    inp = torch.empty(shape, dtype=dtype, device=device)
    key = torch.tensor([12345, 0], dtype=torch.uint64, device=device)
    return inp, key, mean, std


def _native_philox_normal(inp, key, mean=0.0, std=1.0):
    native_op = getattr(torch.ops.aten, "_philox_normal_")
    return native_op(inp, key, mean, std)


@pytest.mark.philox_normal_
@pytest.mark.skipif(
    not _HAS_NATIVE_PHILOX,
    reason="aten::_philox_normal_ is available in PyTorch 2.12 and newer",
)
def test_philox_normal_():
    bench = PhiloxNormalBenchmark(
        op_name="philox_normal_",
        torch_op=_native_philox_normal,
        gems_op=flag_gems.ops._philox_normal_,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
