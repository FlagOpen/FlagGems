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


class MaxPool1dBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (4, 3, 224),
            (16, 64, 128),
            (32, 128, 64),
            (64, 256, 32),
            (128, 512, 16),
        ]


def max_pool1d_case_fn(shape, dtype):
    # Common case
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={
            "kernel_size": [3],
            "stride": [2],
            "padding": [1],
            "dilation": [1],
            "ceil_mode": False,
        },
        builder_args=(shape, [1], False),
    )

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # With dilation
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
                "dilation": [2],
                "ceil_mode": False,
            },
            builder_args=(shape, [2], False),
        )

        # With ceil_mode
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
                "dilation": [1],
                "ceil_mode": True,
            },
            builder_args=(shape, [1], True),
        )


def max_pool1d_build_inputs_fn(plan, dtype, device):
    shape, dilation, ceil_mode = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {
        "kernel_size": [3],
        "stride": [2],
        "padding": [1],
        "dilation": dilation,
        "ceil_mode": ceil_mode,
    }


@pytest.mark.max_pool1d
def test_max_pool1d():
    bench = MaxPool1dBenchmark(
        case_fn=max_pool1d_case_fn,
        build_inputs_fn=max_pool1d_build_inputs_fn,
        op_name="max_pool1d",
        torch_op=torch.ops.aten.max_pool1d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
