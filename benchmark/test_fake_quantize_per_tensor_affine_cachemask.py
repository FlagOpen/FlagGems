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


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={"scale": 0.125, "zero_point": 3, "quant_min": 0, "quant_max": 255},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, 0.125, 3, 0, 255


def _out_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = utils.generate_tensor_input(shape, dtype, device)
    out0 = torch.empty_like(inp)
    out1 = torch.empty_like(inp, dtype=torch.bool)
    return inp, 0.125, 3, 0, 255, {"out0": out0, "out1": out1}


class FakeQuantizePerTensorAffineCachemaskBenchmark(OperatorBenchmark):
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


@pytest.mark.fake_quantize_per_tensor_affine_cachemask
def test_fake_quantize_per_tensor_affine_cachemask():
    bench = FakeQuantizePerTensorAffineCachemaskBenchmark(
        op_name="fake_quantize_per_tensor_affine_cachemask",
        torch_op=torch.ops.aten.fake_quantize_per_tensor_affine_cachemask,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_out
def test_fake_quantize_per_tensor_affine_cachemask_out():
    bench = FakeQuantizePerTensorAffineCachemaskBenchmark(
        op_name="fake_quantize_per_tensor_affine_cachemask_out",
        torch_op=torch.ops.aten.fake_quantize_per_tensor_affine_cachemask.out,
        case_fn=_case_fn,
        build_inputs_fn=_out_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
