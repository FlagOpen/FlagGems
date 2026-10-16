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

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(1024, 1024), (20, 320, 15), (16, 128, 64, 60)]
SCALE = 0.1
ZERO_POINT = 10
Q_DTYPE = torch.quint8


class QuantizeBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "shape"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims)},
        params={"scale": SCALE, "zero_point": ZERO_POINT, "dtype": "quint8"},
        builder_args=(dims,),
    )


def _out_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "out": list(dims)},
        params={"scale": SCALE, "zero_point": ZERO_POINT, "dtype": "quint8"},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    del dtype
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    return torch.randn(shape, dtype=torch.float32, device=device), SCALE, ZERO_POINT, Q_DTYPE


def _out_build_inputs_fn(plan, dtype, device):
    del dtype
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=torch.float32, device=device)
    out = torch._empty_affine_quantized(
        shape, scale=0.5, zero_point=100, dtype=Q_DTYPE, device=device
    )
    return inp, SCALE, ZERO_POINT, Q_DTYPE, {"out": out}


def _torch_out(inp, scale, zero_point, dtype, out=None):
    return torch.ops.aten.quantize_per_tensor.out(
        inp, scale, zero_point, dtype, out=out
    )


@pytest.mark.quantize_per_tensor
def test_quantize_per_tensor():
    QuantizeBenchmark(
        op_name="quantize_per_tensor",
        torch_op=torch.quantize_per_tensor,
        gems_op=flag_gems.quantize_per_tensor,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()


@pytest.mark.quantize_per_tensor_out
def test_quantize_per_tensor_out():
    QuantizeBenchmark(
        op_name="quantize_per_tensor_out",
        torch_op=_torch_out,
        gems_op=flag_gems.quantize_per_tensor_out,
        case_fn=_out_case_fn,
        build_inputs_fn=_out_build_inputs_fn,
        dtypes=[torch.float32],
        is_inplace=True,
        fresh_inputs=True,
    ).run()
