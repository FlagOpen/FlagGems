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


class MaskedSoftmaxBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (64, 64),
            (1024, 1024),
            (4, 8, 512, 512),
            (2, 16, 2048, 2048),
        ]

    def set_more_shapes(self):
        return None


def _case_fn(shape, dtype):
    del dtype
    mask_types = (2,) if len(shape) != 4 else (0, 1, 2)
    for mask_type in mask_types:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"dim": -1, "mask_type": mask_type},
            builder_args=(tuple(shape), mask_type),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, mask_type = plan.builder_args
    torch.manual_seed(0)
    input = utils.generate_tensor_input(shape, dtype, device)
    if mask_type == 0:
        mask_shape = (shape[-1], shape[-1])
    elif mask_type == 1:
        mask_shape = (shape[0], shape[-1])
    else:
        mask_shape = shape
    mask = torch.randint(0, 2, mask_shape, dtype=torch.bool, device=device)
    if mask_type in (0, 1):
        mask[..., 0] = False
    return input, mask, -1, mask_type


@pytest.mark.masked_softmax
def test_masked_softmax():
    bench = MaskedSoftmaxBenchmark(
        op_name="masked_softmax",
        torch_op=torch.ops.aten._masked_softmax,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
