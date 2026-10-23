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

# fbgemm_linear_int8_weight_fp32_activation is delegated to the aten fbgemm
# CPU kernel (the only registered backend). The packed weight object must be
# built on CPU: the fbgemm pack kernel has no CUDA registration and
# moving/cloning a packed tensor drops the PackBMatrix C++ wrapper it carries.
# The native torch baseline (CPU kernel fed CUDA-resident activations)
# segfaults in the fbgemm CPU SIMD unpacking, so the native baseline is
# skipped via the skip_native mark; only the FlagGems wrapper latency is
# measured. Activations are fp32 only (the kernel rejects other dtypes).
PACK_DTYPES = [torch.float32]


class BenchmarkFbgemmLinearInt8WeightFp32Activation(base.Benchmark):
    """
    Benchmark fbgemm_linear_int8_weight_fp32_activation operator
    """

    # Shapes are (M, N): activations [M, N] against an int8 [N, N] weight,
    # so one shape parametrizes both dimensions.
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
            m, n = shape
            weight_fp32 = torch.randn(n, n, dtype=torch.float32)
            weight_int8, col_offsets, weight_scale, weight_zero_point = (
                torch.fbgemm_linear_quantize_weight(weight_fp32)
            )
            packed = torch.ops.aten.fbgemm_pack_quantized_matrix(weight_int8)
            inp = torch.randn(m, n, dtype=torch.float32, device="cuda")
            bias = torch.randn(n, dtype=torch.float32, device="cuda")
            yield inp, weight_int8, packed, col_offsets, weight_scale, weight_zero_point, bias

    def run_torch_op(self, inputs):
        # The packed object must stay host-resident (moving it drops the
        # PackBMatrix wrapper); stage activations/bias like the wrapper does.
        inp, weight_int8, packed, col_offsets, weight_scale, weight_zero_point, bias = (
            inputs
        )
        torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation(
            inp.cpu(),
            weight_int8,
            packed,
            col_offsets,
            weight_scale,
            weight_zero_point,
            bias.cpu(),
        )


@pytest.mark.fbgemm_linear_int8_weight_fp32_activation
@pytest.mark.skip_native(
    vendors=["nvidia"],
    reason="aten::fbgemm_linear_int8_weight_fp32_activation has no CUDA "
    "backend; the native CPU kernel segfaults on CUDA-resident activations",
)
def test_fbgemm_linear_int8_weight_fp32_activation():
    bench = BenchmarkFbgemmLinearInt8WeightFp32Activation(
        op_name="fbgemm_linear_int8_weight_fp32_activation",
        torch_op=torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation,
        gems_op=flag_gems.fbgemm_linear_int8_weight_fp32_activation,
        dtypes=PACK_DTYPES,
    )
    bench.run()
