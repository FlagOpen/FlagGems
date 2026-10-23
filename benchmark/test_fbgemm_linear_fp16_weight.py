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

# fbgemm_linear_fp16_weight is delegated to the aten fbgemm CPU kernel (the
# only registered backend). The packed weight object must be built on CPU:
# the fbgemm pack kernel has no CUDA registration and moving/cloning a packed
# tensor drops the LinearPackedParamsBase C++ wrapper it carries. The native
# torch baseline (CPU kernel fed CUDA-resident activations) segfaults in the
# fbgemm CPU SIMD unpacking, so the native baseline is skipped via the
# skip_native mark; only the FlagGems wrapper latency is measured.
# Activations are fp32 only (the fp16-weight kernel rejects other dtypes).
PACK_DTYPES = [torch.float32]


class BenchmarkFbgemmLinearFp16Weight(base.Benchmark):
    """
    Benchmark fbgemm_linear_fp16_weight operator
    """

    # Shapes are (M, N): activations [M, N] against a packed [N, N] weight,
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
            weight = torch.randn(n, n, dtype=torch.float32)
            packed = torch.ops.aten.fbgemm_pack_gemm_matrix_fp16(weight)
            inp = torch.randn(m, n, dtype=torch.float32, device="cuda")
            bias = torch.randn(n, dtype=torch.float32, device="cuda")
            yield inp, packed, bias


@pytest.mark.fbgemm_linear_fp16_weight
@pytest.mark.skip_native(
    vendors=["nvidia"],
    reason="aten::fbgemm_linear_fp16_weight has no CUDA backend; the native "
    "CPU kernel segfaults on CUDA-resident activations",
)
def test_fbgemm_linear_fp16_weight():
    bench = BenchmarkFbgemmLinearFp16Weight(
        op_name="fbgemm_linear_fp16_weight",
        torch_op=torch.ops.aten.fbgemm_linear_fp16_weight,
        gems_op=flag_gems.fbgemm_linear_fp16_weight,
        dtypes=PACK_DTYPES,
    )
    bench.run()
