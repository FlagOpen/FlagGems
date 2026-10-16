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

SCALE = 0.1
ZERO_POINT = 0
QDTYPE = torch.quint8


def _make_quantized(shape, device):
    torch.manual_seed(0)
    host = torch.quantize_per_tensor(
        torch.randn(shape).clamp_(-2, 2), SCALE, ZERO_POINT, QDTYPE
    )
    result = host if device == "cpu" else host.to(device)
    result._cpu_twin = host
    return result


def _cpu_input(value):
    twin = getattr(value, "_cpu_twin", None)
    return twin if twin is not None else value.to("cpu")

SHAPES = [(4, 3, 16, 56, 56), (8, 64, 8, 28, 28), (16, 128, 4, 14, 14), (32, 256, 2, 7, 7)]
PARAMS = {'kernel_size': [3, 3, 3], 'stride': [2, 2, 2], 'padding': [1, 1, 1], 'dilation': [1, 1, 1], 'ceil_mode': False}


class QuantizedMaxPool3dBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "N, C, " + ", ".join(["D", "H", "W"][-3:])


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    spatial = dims[-3:]
    out_spatial = tuple((size + 2 - 3) // 2 + 1 for size in spatial)
    out_shape = dims[:-3] + out_spatial
    yield base.BenchmarkCasePlan(shape={"self": list(dims), "out": list(out_shape)}, params=dict(PARAMS), builder_args=(dims, out_shape))


def _build_inputs_fn(plan, dtype, device):
    shape, _ = plan.builder_args
    return _make_quantized(shape, device), dict(PARAMS)


def _torch_op(qx, **kwargs):
    return torch.quantized_max_pool3d(_cpu_input(qx), **kwargs)


@pytest.mark.quantized_max_pool3d
def test_quantized_max_pool3d():
    QuantizedMaxPool3dBenchmark(op_name="quantized_max_pool3d", torch_op=_torch_op, gems_op=flag_gems.quantized_max_pool3d, case_fn=_case_fn, build_inputs_fn=_build_inputs_fn, dtypes=[QDTYPE]).run()
