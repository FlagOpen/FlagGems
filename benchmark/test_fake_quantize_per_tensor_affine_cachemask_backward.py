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


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"grad": list(shape), "mask": list(shape)},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    grad = utils.generate_tensor_input(shape, dtype, device)
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    return grad, mask


class FakeQuantizePerTensorAffineCachemaskBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (4, 4),
            (64, 64),
            (128, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
            (8192, 8192),
            (2, 3, 128, 128),
            (8, 16, 64, 64),
            (32, 64, 128, 128),
        ]


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_backward
def test_fake_quantize_per_tensor_affine_cachemask_backward():
    bench = FakeQuantizePerTensorAffineCachemaskBackwardBenchmark(
        op_name="fake_quantize_per_tensor_affine_cachemask_backward",
        torch_op=torch.ops.aten.fake_quantize_per_tensor_affine_cachemask_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
