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

import math

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import set_philox_state

from . import base
from .generated_operator_utils import OperatorBenchmark


class CudnnAttentionForwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (2, 8, 512, 64),
            (4, 16, 1024, 64),
            (2, 16, 2048, 128),
            (1, 8, 4096, 128),
        ]

    def set_more_shapes(self):
        return None


def _case_fn(shape, dtype):
    del dtype
    batch, num_heads, seq_len, head_dim = shape
    for is_causal, has_bias in ((False, False), (True, False), (False, True)):
        yield base.BenchmarkCasePlan(
            shape={
                "query": [batch, num_heads, seq_len, head_dim],
                "key": [batch, num_heads, seq_len, head_dim],
                "value": [batch, num_heads, seq_len, head_dim],
            },
            params={"is_causal": is_causal, "has_attn_bias": has_bias},
            builder_args=(tuple(shape), is_causal, has_bias),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, is_causal, has_bias = plan.builder_args
    batch, num_heads, seq_len, head_dim = shape
    torch.manual_seed(0)
    current = torch_device_fn.current_device()
    set_philox_state(1234567890, 0, current)
    query = torch.empty(shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    key = torch.empty(shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    value = torch.empty(shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    attn_bias = None
    if has_bias:
        attn_bias = torch.randn(
            batch, num_heads, seq_len, seq_len, dtype=dtype, device=device
        ) * 0.1
    return (
        query,
        key,
        value,
        attn_bias,
        None,
        None,
        seq_len,
        seq_len,
        True,
        0.0,
        is_causal,
        False,
        {"scale": 1.0 / math.sqrt(head_dim)},
    )


def _aten(*args, **kwargs):
    return torch.ops.aten._cudnn_attention_forward(*args, **kwargs)


def _gems(*args, **kwargs):
    return flag_gems.ops.cudnn_attention_forward(*args, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cudnn_attention_forward
def test_cudnn_attention_forward():
    bench = CudnnAttentionForwardBenchmark(
        op_name="cudnn_attention_forward",
        torch_op=_aten,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(_gems)
    bench.run()
