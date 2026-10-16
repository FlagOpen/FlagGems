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

SHAPES = [(64, 64, 32), (256, 256, 64), (1024, 1024, 128)]


class DynQuantPack4BitWeightBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "OUT_FEATURES, IN_FEATURES, BLOCK_SIZE"


def _case_fn(shape, dtype):
    del dtype
    out_features, in_features, block_size = shape
    groups = in_features // block_size
    yield base.BenchmarkCasePlan(
        shape={
            "weights": [out_features, in_features // 2],
            "scales_zeros": [out_features, groups],
            "bias": [out_features],
        },
        params={"block_size": block_size},
        builder_args=(out_features, in_features, block_size),
    )


def _build_inputs_fn(plan, dtype, device):
    out_features, in_features, block_size = plan.builder_args
    torch.manual_seed(0)
    weights = torch.randint(
        0,
        256,
        (out_features, in_features // 2),
        dtype=torch.uint8,
        device=device,
    )
    groups = in_features // block_size
    scales = torch.randn((out_features, groups), dtype=dtype, device=device)
    bias = torch.randn(out_features, dtype=dtype, device=device)
    return weights, scales, bias, block_size, in_features, out_features


@pytest.mark.dyn_quant_pack_4bit_weight
@pytest.mark.skipif(
    flag_gems.vendor_name
    in ("nvidia", "hygon", "iluvatar", "ascend", "mthreads", "metax"),
    reason="aten::_dyn_quant_pack_4bit_weight has no native device kernel; CPU fallback is not a performance baseline",
)
def test_dyn_quant_pack_4bit_weight():
    DynQuantPack4BitWeightBenchmark(
        op_name="dyn_quant_pack_4bit_weight",
        torch_op=torch.ops.aten._dyn_quant_pack_4bit_weight.default,
        gems_op=flag_gems._dyn_quant_pack_4bit_weight,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
