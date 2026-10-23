# Copyright 2026, The FlagOS Contributors.
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

from . import base

# _make_per_tensor_quantized_tensor wraps a raw integer payload with a single
# per-tensor scale / zero_point. The payload dtype decides the output quantized
# dtype, so the benchmark always runs the qint8 scheme (int8 payload) -- the
# only dtype in this list (torch.qint8) mirrors the fixed int8 payload; no
# float dtype parametrization applies to this operator.
# Representative quantization shapes: 2D weight/activation tensors and a
# higher-dim activation, kept small enough to avoid OOM.
MPTQ_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
]


def _mptq_input_fn(shape, dtype, device):
    """Yield (payload, scale, zero_point) for the wrap benchmark."""
    # The payload is always int8 (qint8 scheme); INT_DTYPES would parametrize
    # over wider ints whose quantized schemes have no QuantizedCUDA kernel.
    payload = torch.randint(-100, 100, shape, device=device, dtype=torch.int8)
    yield payload, 0.05, 10


class MakePerTensorQuantizedTensorBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = MPTQ_SHAPES

    def set_more_shapes(self):
        return []


@pytest.mark.make_per_tensor_quantized_tensor
def test_make_per_tensor_quantized_tensor():
    bench = MakePerTensorQuantizedTensorBenchmark(
        op_name="make_per_tensor_quantized_tensor",
        torch_op=torch.ops.aten._make_per_tensor_quantized_tensor,
        input_fn=_mptq_input_fn,
        dtypes=[torch.qint8],
    )
    bench.run()


def _mptq_out_input_fn(shape, dtype, device):
    """Yield (payload, scale, zero_point, {'out': out})."""
    payload = torch.randint(-100, 100, shape, device=device, dtype=torch.int8)
    # Pre-allocate the per-tensor quantized ``out`` through the public
    # quantize_per_tensor factory on a zeros tensor; the FlagGems out kernel
    # overwrites its integer storage and adopts the source quantizer
    # (quantized copy_ semantics).
    out = torch.quantize_per_tensor(
        torch.zeros(shape, dtype=torch.float32, device=device),
        0.9,
        3,
        torch.qint8,
    )
    yield payload, 0.05, 10, {"out": out}


class MakePerTensorQuantizedTensorOutBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = MPTQ_SHAPES

    def set_more_shapes(self):
        return []


@pytest.mark.make_per_tensor_quantized_tensor_out
def test_make_per_tensor_quantized_tensor_out():
    bench = MakePerTensorQuantizedTensorOutBenchmark(
        op_name="make_per_tensor_quantized_tensor_out",
        torch_op=torch.ops.aten._make_per_tensor_quantized_tensor.out,
        input_fn=_mptq_out_input_fn,
        dtypes=[torch.qint8],
    )
    bench.run()
