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

# fbgemm_linear_quantize_weight is delegated to the aten fbgemm CPU kernel
# (the only registered backend). The kernel reads the weight payload with
# plain host pointers, so the native torch baseline fed a CUDA-resident
# weight segfaults in the fbgemm CPU SIMD code; the native baseline is
# skipped via the skip_native mark and only the FlagGems wrapper latency is
# measured. Weights are fp32 only (the kernel rejects other dtypes).
PACK_DTYPES = [torch.float32]


class BenchmarkFbgemmLinearQuantizeWeight(base.Benchmark):
    """
    Benchmark fbgemm_linear_quantize_weight operator
    """

    # Shapes are the linear-weight quantization sizes (N_out, K): each is a
    # single fp32 [N, K] weight, matching the fbgemm int8 linear use case.
    DEFAULT_SHAPES = [
        (64, 64),
        (256, 256),
        (1024, 1024),
        (1024, 512),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            weight = torch.randn(shape, dtype=torch.float32, device="cuda")
            yield (weight,)

    def run_torch_op(self, inputs):
        # The aten fbgemm kernel runs on the host; stage the weight like the
        # wrapper does to avoid the fbgemm SIMD segfault on device pointers.
        (weight,) = inputs
        torch.ops.aten.fbgemm_linear_quantize_weight(weight.cpu())


@pytest.mark.fbgemm_linear_quantize_weight
@pytest.mark.skip_native(
    vendors=["nvidia"],
    reason="aten::fbgemm_linear_quantize_weight has no CUDA backend; the "
    "native CPU kernel segfaults on CUDA-resident weights",
)
def test_fbgemm_linear_quantize_weight():
    bench = BenchmarkFbgemmLinearQuantizeWeight(
        op_name="fbgemm_linear_quantize_weight",
        torch_op=torch.ops.aten.fbgemm_linear_quantize_weight,
        gems_op=flag_gems.fbgemm_linear_quantize_weight,
        dtypes=PACK_DTYPES,
    )
    bench.run()
