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
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(64, 64), (256, 256), (1024, 1024), (4096, 4096)]
NUM_PARAMS = 4


class FusedSgdBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "optimizer parameter shape"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for mode, weight_decay, dampening, nesterov, maximize in (
        ("momentum", 0.0, 0.0, False, False),
        ("full", 0.01, 0.2, True, True),
    ):
        yield base.BenchmarkCasePlan(
            shape={"param": list(dims), "count": NUM_PARAMS},
            params={
                "mode": mode,
                "weight_decay": weight_decay,
                "dampening": dampening,
                "nesterov": nesterov,
                "maximize": maximize,
            },
            builder_args=(dims, weight_decay, dampening, nesterov, maximize),
        )


def _tensor_lr_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"param": list(dims), "count": NUM_PARAMS},
        params={"lr_kind": "tensor", "momentum": 0.9},
        builder_args=(dims,),
    )


def _build_common(shape, dtype, device):
    torch.manual_seed(0)
    params = [
        torch.randn(shape, dtype=dtype, device=device) for _ in range(NUM_PARAMS)
    ]
    grads = [torch.randn_like(param) for param in params]
    momentum_bufs = [torch.randn_like(param) for param in params]
    return params, grads, momentum_bufs


def _build_inputs_fn(plan, dtype, device):
    shape, weight_decay, dampening, nesterov, maximize = plan.builder_args
    params, grads, momentum_bufs = _build_common(shape, dtype, device)
    kwargs = {
        "weight_decay": weight_decay,
        "momentum": 0.9,
        "lr": 0.1,
        "dampening": dampening,
        "nesterov": nesterov,
        "maximize": maximize,
        "is_first_step": False,
        "grad_scale": None,
        "found_inf": None,
    }
    return params, grads, momentum_bufs, kwargs


def _tensor_lr_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    params, grads, momentum_bufs = _build_common(shape, dtype, device)
    kwargs = {
        "weight_decay": 0.01,
        "momentum": 0.9,
        "lr": torch.tensor(0.1, dtype=torch.float32, device=device),
        "dampening": 0.0,
        "nesterov": True,
        "maximize": False,
        "is_first_step": False,
        "grad_scale": None,
        "found_inf": None,
    }
    return params, grads, momentum_bufs, kwargs


@pytest.mark.fused_sgd_
def test_fused_sgd_():
    FusedSgdBenchmark(
        op_name="fused_sgd_",
        torch_op=torch._fused_sgd_,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        fresh_inputs=True,
    ).run()


@pytest.mark.fused_sgd_
def test_fused_sgd__tensor_lr():
    FusedSgdBenchmark(
        op_name="fused_sgd__tensor_lr",
        torch_op=torch.ops.aten._fused_sgd_.tensor_lr,
        case_fn=_tensor_lr_case_fn,
        build_inputs_fn=_tensor_lr_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        fresh_inputs=True,
    ).run()
