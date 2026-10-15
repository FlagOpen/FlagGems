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


@pytest.mark.bitwise_xor_tensor
def test_bitwise_xor_tensor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_xor_tensor",
        torch_op=torch.bitwise_xor,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.bitwise_xor_tensor_
def test_bitwise_xor_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_xor_tensor_",
        torch_op=lambda a, b: a.bitwise_xor_(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _scalar_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        builder_args=(shape,),
    )


def _scalar_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    scalar = True if dtype == torch.bool else 0x00FF
    return inp, scalar


@pytest.mark.bitwise_xor_scalar
def test_bitwise_xor_scalar():
    bench = base.GenericBenchmark(
        op_name="bitwise_xor_scalar",
        torch_op=torch.bitwise_xor,
        case_fn=_scalar_case_fn,
        build_inputs_fn=_scalar_build_inputs_fn,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


def _scalar_build_inputs_fn_inplace(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    scalar = True if dtype == torch.bool else 0x5A
    return inp, scalar


@pytest.mark.bitwise_xor_scalar_
def test_bitwise_xor_scalar_():
    bench = base.GenericBenchmark(
        case_fn=_scalar_case_fn,
        build_inputs_fn=_scalar_build_inputs_fn_inplace,
        op_name="bitwise_xor_scalar_",
        torch_op=lambda a, b: a.bitwise_xor_(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()


def scalar_tensor_build_inputs_fn(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    scalar = 0x00FF if cur_dtype != torch.bool else True
    tensor = utils.generate_tensor_input(shape, cur_dtype, device)
    return scalar, tensor


@pytest.mark.bitwise_xor_scalar_tensor
def test_bitwise_xor_scalar_tensor():
    bench = base.GenericBenchmark(
        op_name="bitwise_xor_scalar_tensor",
        torch_op=torch.bitwise_xor,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        case_fn=_scalar_case_fn,
        build_inputs_fn=scalar_tensor_build_inputs_fn,
    )
    bench.run()
