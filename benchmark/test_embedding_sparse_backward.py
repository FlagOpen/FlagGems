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

SHAPES = [(32, 2048, 128, 8192), (16, 2048, 256, 16384), (8, 4096, 256, 32768)]
PADDING_INDICES = (-1, 0, 5)


class EmbeddingSparseBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "B, M, D, num_weights"


def _case_fn(shape, dtype):
    del dtype
    b, m, d, num_weights = tuple(shape)
    for padding_idx in PADDING_INDICES:
        yield base.BenchmarkCasePlan(
            shape={"grad": [b, m, d], "indices": [b, m]},
            params={"num_weights": num_weights, "padding_idx": padding_idx, "scale_grad_by_freq": False},
            builder_args=(b, m, d, num_weights, padding_idx),
        )


def _build_inputs_fn(plan, dtype, device):
    b, m, d, num_weights, padding_idx = plan.builder_args
    torch.manual_seed(0)
    grad = torch.randn((b, m, d), dtype=dtype, device=device)
    indices = torch.randint(0, num_weights, (b, m), dtype=torch.long, device=device)
    if padding_idx >= 0:
        mask = torch.rand((b, m), device=device) < 0.25
        indices = torch.where(mask, torch.full_like(indices, padding_idx), indices)
    return grad, indices, num_weights, padding_idx, False


@pytest.mark.embedding_sparse_backward
def test_embedding_sparse_backward():
    EmbeddingSparseBackwardBenchmark(
        op_name="embedding_sparse_backward",
        torch_op=torch.ops.aten.embedding_sparse_backward,
        gems_op=flag_gems.embedding_sparse_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
