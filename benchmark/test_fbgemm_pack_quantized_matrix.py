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

# fbgemm_pack_quantized_matrix is delegated to the aten fbgemm CPU kernel
# (the only registered backend). It returns an opaque packed tensor whose
# payload carries an fbgemm C++ wrapper object; the packed layout is
# fbgemm-private, so no numerical baseline exists. The native torch baseline
# (CPU kernel fed a CUDA-resident weight) segfaults in the fbgemm CPU SIMD
# packing, so the native baseline is skipped via the skip_native mark; only
# the FlagGems wrapper latency is measured. Weights are int8 only (the
# kernel rejects other dtypes).
# INT_DTYPES is not usable directly: this op takes int8 weights only
# (fp16/fp32/int64/uint8 are rejected), so the matrix is fixed instead of
# parametrizing over the full integer dtype list.
PACK_INT_DTYPES = [torch.int8]


class BenchmarkFbgemmPackQuantizedMatrix(base.Benchmark):
    """
    Benchmark fbgemm_pack_quantized_matrix operator
    """

    # Shapes are (N_out, K): the int8 weight matrix [N_out, K] to be packed.
    # The pack cost scales with K * N_out (SIMD blocking of the matrix), so
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
            weight = torch.randint(-128, 127, shape, dtype=torch.int8, device="cuda")
            yield (weight,)


@pytest.mark.fbgemm_pack_quantized_matrix
@pytest.mark.skip_native(
    vendors=["nvidia"],
    reason="aten::fbgemm_pack_quantized_matrix has no CUDA backend; the "
    "native CPU kernel segfaults on CUDA-resident weights",
)
def test_fbgemm_pack_quantized_matrix():
    bench = BenchmarkFbgemmPackQuantizedMatrix(
        op_name="fbgemm_pack_quantized_matrix",
        torch_op=torch.ops.aten.fbgemm_pack_quantized_matrix,
        gems_op=flag_gems.fbgemm_pack_quantized_matrix,
        dtypes=PACK_INT_DTYPES,
    )
    bench.run()
