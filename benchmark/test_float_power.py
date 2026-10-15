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


class FloatPowerBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(4,), (1024,), (65536,), (512, 512), (2048, 2048)]


def _simple_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        builder_args=(shape,),
    )


def _tensor_tensor_build(plan, dtype, device):
    (shape,) = plan.builder_args
    base_tensor = torch.rand(shape, dtype=dtype, device=device).add_(0.25)
    exponent = torch.rand(shape, dtype=dtype, device=device).mul_(4).sub_(2)
    return base_tensor, exponent


def _tensor_scalar_build(plan, dtype, device):
    (shape,) = plan.builder_args
    base_tensor = torch.rand(shape, dtype=dtype, device=device).add_(0.25)
    return base_tensor, 1.234


def _scalar_tensor_build(plan, dtype, device):
    (shape,) = plan.builder_args
    exponent = torch.rand(shape, dtype=dtype, device=device).mul_(4).sub_(2)
    return 2.0, exponent


def _tensor_tensor_out_build(plan, dtype, device):
    (shape,) = plan.builder_args
    base_tensor = torch.rand(shape, dtype=dtype, device=device).add_(0.25)
    exponent = torch.rand(shape, dtype=dtype, device=device).mul_(4).sub_(2)
    out = torch.empty(shape, dtype=torch.float64, device=device)
    return base_tensor, exponent, {"out": out}


def _tensor_scalar_out_build(plan, dtype, device):
    (shape,) = plan.builder_args
    base_tensor = torch.rand(shape, dtype=dtype, device=device).add_(0.25)
    out = torch.empty(shape, dtype=torch.float64, device=device)
    return base_tensor, 1.234, {"out": out}


def _scalar_tensor_out_build(plan, dtype, device):
    (shape,) = plan.builder_args
    exponent = torch.rand(shape, dtype=dtype, device=device).mul_(4).sub_(2)
    out = torch.empty(shape, dtype=torch.float64, device=device)
    return 2.0, exponent, {"out": out}


def _run_benchmark(op_name, build_inputs_fn, torch_op):
    bench = FloatPowerBenchmark(
        case_fn=_simple_case_fn,
        build_inputs_fn=build_inputs_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.float_power_tensor_tensor
def test_float_power_tensor_tensor():
    _run_benchmark(
        "float_power_tensor_tensor",
        _tensor_tensor_build,
        torch.ops.aten.float_power.Tensor_Tensor,
    )


@pytest.mark.float_power_tensor_scalar
def test_float_power_tensor_scalar():
    _run_benchmark(
        "float_power_tensor_scalar",
        _tensor_scalar_build,
        torch.ops.aten.float_power.Tensor_Scalar,
    )


@pytest.mark.float_power_scalar_tensor
def test_float_power_scalar_tensor():
    _run_benchmark(
        "float_power_scalar_tensor",
        _scalar_tensor_build,
        torch.ops.aten.float_power.Scalar,
    )


@pytest.mark.float_power_tensor_tensor_out
def test_float_power_tensor_tensor_out():
    _run_benchmark(
        "float_power_tensor_tensor_out",
        _tensor_tensor_out_build,
        torch.ops.aten.float_power.Tensor_Tensor_out,
    )


@pytest.mark.float_power_tensor_scalar_out
def test_float_power_tensor_scalar_out():
    _run_benchmark(
        "float_power_tensor_scalar_out",
        _tensor_scalar_out_build,
        torch.ops.aten.float_power.Tensor_Scalar_out,
    )


@pytest.mark.float_power_scalar_tensor_out
def test_float_power_scalar_tensor_out():
    _run_benchmark(
        "float_power_scalar_tensor_out",
        _scalar_tensor_out_build,
        torch.ops.aten.float_power.Scalar_out,
    )
