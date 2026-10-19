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

SHAPES = [(16, 8, 64), (32, 16, 128), (64, 32, 256), (128, 64, 256)]


class PackPaddedSequenceBenchmark(OperatorBenchmark):
    DEFAULT_SHAPES = SHAPES


def _case_fn(shape, dtype):
    del dtype
    time_steps, batch_size, features = shape
    yield base.BenchmarkCasePlan(
        shape={"input": [time_steps, batch_size, features], "lengths": [batch_size]},
        params={"batch_first": False},
        builder_args=(time_steps, batch_size, features),
    )


def _build_inputs_fn(plan, dtype, device):
    time_steps, batch_size, features = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(time_steps, batch_size, features, dtype=dtype, device=device)
    lengths = torch.linspace(time_steps, 1, steps=batch_size, dtype=torch.int64).clamp(min=1)
    lengths = torch.sort(lengths, descending=True).values
    return inp, lengths, False


@pytest.mark.pack_padded_sequence
def test_pack_padded_sequence():
    PackPaddedSequenceBenchmark(
        op_name="pack_padded_sequence",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._pack_padded_sequence,
        gems_op=getattr(flag_gems, "pack_padded_sequence", None),
        dtypes=consts.FLOAT_DTYPES,
    ).run()
