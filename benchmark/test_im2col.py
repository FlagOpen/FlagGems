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

IM2COL_SHAPES_4D = [(1, 3, 16, 16), (1, 3, 32, 32), (2, 16, 64, 64), (4, 32, 128, 128)]
IM2COL_CONFIGS = [
    ((3, 3), (1, 1), (1, 1), (1, 1)),
    ((3, 3), (1, 1), (0, 0), (2, 2)),
    ((5, 4), (2, 2), (2, 1), (1, 2)),
    ((1, 1), (1, 1), (0, 0), (1, 1)),
]


class Im2colBenchmark(base.Benchmark):
    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = IM2COL_SHAPES_4D

    def get_case_iter(self, cur_dtype):
        ordinal = 0
        for shape in self.shapes:
            for kernel_size, dilation, padding, stride in IM2COL_CONFIGS:
                yield self._case_from_plan(
                    cur_dtype,
                    ordinal,
                    base.BenchmarkCasePlan(
                        shape={"input": list(shape)},
                        params={
                            "kernel_size": kernel_size,
                            "dilation": dilation,
                            "padding": padding,
                            "stride": stride,
                        },
                        builder_args=(shape, kernel_size, dilation, padding, stride),
                    ),
                )
                ordinal += 1

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, kernel_size, dilation, padding, stride = plan.builder_args
        x = torch.randn(shape, dtype=case.dtype, device=self.device)
        return x, kernel_size, dilation, padding, stride


@pytest.mark.im2col
def test_im2col():
    bench = Im2colBenchmark(
        op_name="im2col",
        torch_op=torch.ops.aten.im2col,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
