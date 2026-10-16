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

from . import base, consts, utils


SHAPES = [
    (4, 3, 16, 56, 56),
    (8, 64, 8, 28, 28),
    (16, 128, 4, 14, 14),
    (32, 256, 2, 7, 7),
]


def _case_fn(shape, dtype):
    del dtype
    configs = [
        {
            "kernel_size": [3, 3, 3],
            "stride": [2, 2, 2],
            "padding": [1, 1, 1],
            "dilation": [1, 1, 1],
            "ceil_mode": False,
        }
    ]
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        if min(shape[-3:]) > 5:
            configs.append(
                {
                    "kernel_size": [2, 3, 3],
                    "stride": [1, 2, 2],
                    "padding": [0, 1, 1],
                    "dilation": [1, 1, 1],
                    "ceil_mode": False,
                }
            )
        configs.append(
            {
                "kernel_size": [3, 3, 3],
                "stride": [2, 2, 2],
                "padding": [1, 1, 1],
                "dilation": [1, 1, 1],
                "ceil_mode": True,
            }
        )
    for params in configs:
        yield base.BenchmarkCasePlan(
            shape={"self": list(shape)},
            params=params,
            builder_args=(tuple(shape), params),
        )


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(0)
    shape, params = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, params


class MaxPool3DBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES
        self.shape_desc = "N, C, D, H, W"


@pytest.mark.max_pool3d
def test_max_pool3d():
    MaxPool3DBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="max_pool3d",
        torch_op=torch.max_pool3d,
        gems_op=flag_gems.max_pool3d,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
