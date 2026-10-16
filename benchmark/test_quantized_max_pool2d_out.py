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

SHAPES = [(4, 3, 224, 224), (16, 64, 56, 56), (32, 128, 28, 28), (64, 256, 14, 14), (128, 512, 7, 7)]
PARAMS = {'kernel_size': [3, 3], 'stride': [2, 2], 'padding': [1, 1], 'dilation': [1, 1], 'ceil_mode': False}


class QuantizedMaxPool2dBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "N, C, " + ", ".join(["D", "H", "W"][-2:])


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    spatial = dims[-2:]
    out_spatial = tuple((size + 2 - 3) // 2 + 1 for size in spatial)
    out_shape = dims[:-2] + out_spatial
    yield base.BenchmarkCasePlan(shape={"self": list(dims), "out": list(out_shape)}, params=dict(PARAMS), builder_args=(dims, out_shape))


def _build_inputs_fn(plan, dtype, device):
    shape, out_shape = plan.builder_args
    qx = _make_quantized(shape, device)
    out = torch._empty_affine_quantized(out_shape, scale=SCALE, zero_point=ZERO_POINT, dtype=QDTYPE, device=device)
    out._cpu_twin = torch._empty_affine_quantized(out_shape, scale=SCALE, zero_point=ZERO_POINT, dtype=QDTYPE, device="cpu")
    return qx, {**PARAMS, "out": out}


def _torch_op(qx, **kwargs):
    out = kwargs.pop("out")
    return torch.ops.aten.quantized_max_pool2d.out(_cpu_input(qx), out=_cpu_input(out), **kwargs)


def _gems_op(qx, **kwargs):
    return flag_gems.quantized_max_pool2d_out(qx, **kwargs)


@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out():
    QuantizedMaxPool2dBenchmark(op_name="quantized_max_pool2d_out", torch_op=_torch_op, gems_op=_gems_op, case_fn=_case_fn, build_inputs_fn=_build_inputs_fn, dtypes=[QDTYPE], is_inplace=True, fresh_inputs=True).run()
