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

from . import base
from .generated_operator_utils import OperatorBenchmark

FORWARD = torch.ops.aten._scaled_dot_product_efficient_attention
BACKWARD = torch.ops.aten._scaled_dot_product_efficient_attention_backward
SHAPES = [(1, 2, 64, 64), (2, 4, 128, 64)]
DTYPES = [torch.float16, torch.bfloat16]


class AttentionBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "(batch, heads, sequence, head_dim)"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for is_causal in (False, True):
        yield base.BenchmarkCasePlan(
            shape={"grad_out": list(dims), "query": list(dims)},
            params={"is_causal": is_causal, "grad_input_mask": [True, True, True, False]},
            builder_args=(dims, is_causal),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, is_causal = plan.builder_args
    torch.manual_seed(0)
    query = torch.randn(shape, dtype=dtype, device=device)
    key = torch.randn(shape, dtype=dtype, device=device)
    value = torch.randn(shape, dtype=dtype, device=device)
    grad_out = torch.empty(shape, dtype=dtype, device=device).normal_()
    scale = 1.0 / (shape[-1] ** 0.5)
    forward = FORWARD(query, key, value, None, True, 0.0, is_causal, scale=scale)
    return (
        grad_out, query, key, value, None, forward[0], forward[1], forward[2],
        forward[3], 0.0, [True, True, True, False], is_causal, {"scale": scale},
    )


@pytest.mark.scaled_dot_product_efficient_attention_backward
def test_scaled_dot_product_efficient_attention_backward():
    AttentionBackwardBenchmark(
        op_name="scaled_dot_product_efficient_attention_backward",
        torch_op=BACKWARD,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=DTYPES,
    ).run()
