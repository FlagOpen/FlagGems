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

# fbgemm_pack_gemm_matrix_fp16 is delegated to the aten fbgemm CPU kernel
# (the only registered backend). It returns an opaque packed tensor whose
# payload carries an fbgemm C++ wrapper object; the packed layout is
# fbgemm-private, so no numerical baseline exists. The native torch baseline
# (CPU kernel fed a CUDA-resident weight) segfaults in the fbgemm CPU SIMD
# packing, so the native baseline is skipped via the skip_native mark; only
# the FlagGems wrapper latency is measured. Weights are fp32 only (the
# kernel rejects other dtypes).
PACK_DTYPES = [torch.float32]


class BenchmarkFbgemmPackGemmMatrixFp16(base.Benchmark):
    """
    Benchmark fbgemm_pack_gemm_matrix_fp16 operator
    """

    # Shapes are (N_out, K): the weight matrix [N_out, K] to be packed. The
    # pack cost scales with K * N_out (fp16 SIMD blocking of the matrix), so
    # square and skewed matrix sizes are both measured. Fixed list because
    # the op has no natural runtime shape parameter beyond the weight size.
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


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
@pytest.mark.skip_native(
    vendors=["nvidia"],
    reason="aten::fbgemm_pack_gemm_matrix_fp16 has no CUDA backend; the "
    "native CPU kernel segfaults on CUDA-resident weights",
)
def test_fbgemm_pack_gemm_matrix_fp16():
    bench = BenchmarkFbgemmPackGemmMatrixFp16(
        op_name="fbgemm_pack_gemm_matrix_fp16",
        torch_op=torch.ops.aten.fbgemm_pack_gemm_matrix_fp16,
        gems_op=flag_gems.fbgemm_pack_gemm_matrix_fp16,
        dtypes=PACK_DTYPES,
    )
    bench.run()
