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
"""Tests for fbgemm_linear_fp16_weight_fp32_activation (delegated implementation).

Reference strategy
------------------
``fbgemm_linear_fp16_weight_fp32_activation`` is delegated to aten (there is
no Triton kernel). ``packed_weight`` is an opaque CPU tensor whose TensorImpl
carries an fbgemm ``LinearPackedParamsBase`` C++ object; the packing format is
owned by the fbgemm library and cannot be synthesized by FlagGems. On this
build the op only has a CPU kernel, and the CUDA-resident path of native aten
segfaults (the fbgemm CPU SIMD unpacking dereferences device pointers), so
the FlagGems wrapper stages CUDA activations through host memory around the
same aten call.

The tests therefore validate:

* the numerical contract against a **logical-equivalent, fbgemm-free
  reference**: unpack semantics of the fp16 packed weight equal a plain
  ``w.half()`` round-trip, so the reference is
  ``torch.nn.functional.linear(input, weight.half().float(), bias)`` with
  fp32 activations (the kernel converts them to fp16 internally);
* the wrapper's validation logic (dtype / dim / error paths) via illegal
  inputs;
* device routing: CUDA inputs flow through the wrapper and produce results
  equal to the CPU reference, where the native aten call itself crashes.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# fbgemm_linear_fp16_weight_fp32_activation accepts fp32 activations only
# (fp16/fp64/bf16 are rejected by the aten fp16-weight kernel) and produces
# fp32 output, so the dtype matrix is fixed instead of parametrizing over
# FLOAT_DTYPES.
SHAPES = (
    [(16, 32), (64, 128)]
    if utils.QUICK_MODE
    else [
        (8, 16),
        (64, 128),
        (512, 512),
        (16, 128, 64),
    ]
)

# The packed weight must be built on CPU: the fbgemm pack kernel has no CUDA
# registration, and moving a packed tensor to CUDA (or cloning it) drops the
# LinearPackedParamsBase wrapper stored in the TensorImpl.
N_OUT, N_IN = 16, 32


def _packed_weight(n_out=N_OUT, n_in=N_IN, seed=0):
    """Build a real fbgemm-packed fp16 weight object (CPU-resident)."""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(n_out, n_in, generator=generator, dtype=torch.float32)
    packed = torch.ops.aten.fbgemm_pack_gemm_matrix_fp16(weight)
    return weight, packed


def _reference(input, weight, bias):
    """fbgemm-free logical equivalent: fp32 activation, fp16-rounded weight."""
    weight = weight.to(input.device)
    if bias is not None:
        bias = bias.to(input.device)
    return torch.nn.functional.linear(input, weight.half().float(), bias)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
@pytest.mark.parametrize("shape", SHAPES)
def test_fbgemm_linear_fp16_weight_fp32_activation(shape):
    weight, packed = _packed_weight(n_in=shape[-1])
    input = torch.randn(shape, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight.shape[0], device=flag_gems.device)

    res_out = flag_gems.fbgemm_linear_fp16_weight_fp32_activation(input, packed, bias)

    # The FlagGems wrapper executes the aten fbgemm CPU kernel (host staging
    # for CUDA activations), while the logical reference runs on the device
    # through cublas. Both round the weight to fp16 first; only the fp32
    # accumulation order differs, so compare with a tolerance that covers
    # GEMM reduction-order noise.
    ref_out = _reference(utils.to_reference(input), weight, utils.to_reference(bias))
    utils.gems_assert_close(res_out, ref_out, dtype=torch.float32, atol=2e-2)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
def test_fbgemm_linear_fp16_weight_fp32_activation_fp16_bias_upcast():
    """A fp16 bias is accepted and upcast: out == linear(x, unpack(w), b)."""
    weight, packed = _packed_weight(seed=1)
    input = torch.randn(8, N_IN, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight.shape[0], device=flag_gems.device, dtype=torch.float16)

    res_out = flag_gems.fbgemm_linear_fp16_weight_fp32_activation(input, packed, bias)
    ref_out = _reference(
        utils.to_reference(input), weight, utils.to_reference(bias).float()
    )
    utils.gems_assert_close(res_out, ref_out, dtype=torch.float32, atol=2e-2)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.bfloat16])
def test_fbgemm_linear_fp16_weight_fp32_activation_unsupported_activation(dtype):
    """Non-fp32 activations are rejected before dispatch, like aten."""
    weight, packed = _packed_weight()
    input = torch.randn(4, N_IN, device=flag_gems.device, dtype=dtype)
    bias = torch.randn(weight.shape[0], device=flag_gems.device)
    with pytest.raises(RuntimeError, match="expected scalar type Float"):
        flag_gems.fbgemm_linear_fp16_weight_fp32_activation(input, packed, bias)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
def test_fbgemm_linear_fp16_weight_fp32_activation_rejects_1d_input():
    """A vector input has no matrix dimension: rejected like aten."""
    weight, packed = _packed_weight()
    input = torch.randn(N_IN, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight.shape[0], device=flag_gems.device)
    with pytest.raises(RuntimeError, match=r"input\.dim\(\) >= 2"):
        flag_gems.fbgemm_linear_fp16_weight_fp32_activation(input, packed, bias)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
def test_fbgemm_linear_fp16_weight_fp32_activation_k_mismatch():
    """An input whose last dim differs from the packed K raises."""
    weight, packed = _packed_weight()
    input = torch.randn(8, N_IN + 4, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight.shape[0], device=flag_gems.device)
    with pytest.raises(RuntimeError, match="numRows"):
        flag_gems.fbgemm_linear_fp16_weight_fp32_activation(input, packed, bias)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
def test_fbgemm_linear_fp16_weight_fp32_activation_non_contiguous_input():
    """A strided activation is accepted and matches the contiguous result."""
    weight, packed = _packed_weight()
    pool = torch.randn(8, N_IN * 2, device=flag_gems.device, dtype=torch.float32)
    input = pool[:, ::2]
    bias = torch.randn(weight.shape[0], device=flag_gems.device)

    res_out = flag_gems.fbgemm_linear_fp16_weight_fp32_activation(input, packed, bias)
    ref_out = flag_gems.fbgemm_linear_fp16_weight_fp32_activation(
        input.contiguous(), packed, bias
    )
    # Both sides come from the same wrapper; route through to_reference so the
    # --ref cpu device convention holds for the shared assert helper.
    utils.gems_assert_equal(utils.to_reference(res_out), utils.to_reference(ref_out))
