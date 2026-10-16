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

INNER_SHAPES = [
    ((4096,), (4096,)),
    ((16384,), (16384,)),
    ((512, 1024), (1024, 1024)),
    ((1024, 4096), (2048, 4096)),
    ((2048, 4096), (2048, 4096)),
    ((256, 128), (128,)),
]


def _case_fn(shape, dtype):
    del dtype
    input_shape, other_shape = shape
    yield base.BenchmarkCasePlan(
        shape={"input": list(input_shape), "other": list(other_shape)},
        params={},
        builder_args=(input_shape, other_shape),
    )


def _build_inputs_fn(plan, dtype, device):
    input_shape, other_shape = plan.builder_args
    torch.manual_seed(0)
    inp = utils.generate_tensor_input(input_shape, dtype, device)
    other = utils.generate_tensor_input(other_shape, dtype, device)
    return inp, other


class InnerBenchmark(OperatorBenchmark):
    DEFAULT_SHAPE_DESC = "input shape, other shape"

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=INNER_SHAPES)


@pytest.mark.inner
def test_inner():
    bench = InnerBenchmark(
        op_name="inner",
        torch_op=torch.inner,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
