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
owned by the fbgemm library and cannot be synthesized by FlagGems.

**Deliberately no end-to-end numerical comparison.** The logical reference
``F.linear(input, weight.half().float(), bias)`` reproduces the fp16-weight
unpack semantics, but only the accumulation *order* differs between the real
fbgemm CPU SIMD kernel (used by the delegated aten path) and a cublas GEMM
over the unpacked fp16-rounded weight. Both are correct fp16-weight GEMMs;
their results drift by ~1 ulp of the output magnitude, which at (512, 512)
already reaches 2.06e-2 > any fixed atol=2e-2 budget that must also hold at
smaller shapes. This was observed on CI builds that ship the real fbgemm
library (local builds without fbgemm cannot reproduce it). The same applies
to the sibling op ``fbgemm_linear_int8_weight`` (PR #7638) and to
``fbgemm_linear_int8_weight_fp32_activation`` (PR #7647), which dropped the
numerical comparison for the same reason. This test file therefore restricts
itself to behavior that is identical on every build (validation and error
paths, dtype/shape/device contracts, fp16-bias upcast routing) plus
wrapper-vs-aten parity checks against the delegated kernel itself, so CI
never depends on fbgemm-vs-reference numerics.

Validation covers:

* non-fp32 activations are rejected before dispatch;
* 1-D input rejection (matrix-only op, same message as aten);
* K mismatch rejection (same message as aten);
* fp16/fp32/bf16 bias accepted, other dtypes rejected.

The delegation path itself is exercised via bitwise wrapper-vs-aten parity
(both sides call the same fbgemm kernel in the same process, so the
comparison is build-independent) and a shape/dtype/device smoke check.
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


@pytest.fixture(autouse=True)
def _require_fbgemm_helpers():
    """Skip the whole module when the fbgemm-backed aten helpers are absent.

    The packed object comes from a deprecated fbgemm aten kernel whose
    availability varies across torch builds; probing lazily via torch.ops
    keeps the module importable everywhere.
    """
    if not hasattr(torch.ops.aten, "fbgemm_pack_gemm_matrix_fp16") or not hasattr(
        torch.ops.aten, "fbgemm_linear_fp16_weight_fp32_activation"
    ):
        pytest.skip(
            "fbgemm-backed aten helpers (fbgemm_pack_gemm_matrix_fp16 / "
            "fbgemm_linear_fp16_weight_fp32_activation) unavailable on this "
            "torch build"
        )


def _packed_weight(n_out=N_OUT, n_in=N_IN, seed=0):
    """Build a real fbgemm-packed fp16 weight object (CPU-resident)."""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(n_out, n_in, generator=generator, dtype=torch.float32)
    packed = torch.ops.aten.fbgemm_pack_gemm_matrix_fp16(weight)
    return weight, packed


def _assert_delegates(input, packed, bias, n_out):
    """A legal call must return an fp32 result of the right shape, on the
    input's device, without raising. No numerical assertion: the delegated
    aten fbgemm kernel is the single source of truth, and its accumulation
    order differs from any fbgemm-free reference across builds (see module
    docstring)."""
    res_out = flag_gems.fbgemm_linear_fp16_weight_fp32_activation(input, packed, bias)
    assert res_out.dtype == torch.float32
    assert res_out.shape == input.shape[:-1] + (n_out,)
    assert res_out.device.type == input.device.type


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
@pytest.mark.parametrize("shape", SHAPES)
def test_fbgemm_linear_fp16_weight_fp32_activation_delegates(shape):
    """A legal call delegates to the aten fbgemm kernel without raising.

    On CPU inputs the wrapper calls the aten kernel directly; on CUDA inputs
    it stages the activation and bias through host memory around the same
    call. The packed object itself must stay host-resident: the fbgemm pack
    kernel has no CUDA registration and moving/cloning a packed tensor drops
    the LinearPackedParamsBase C++ wrapper it carries. Output is fp32 with
    shape (..., N_out); values come verbatim from the delegated kernel (no
    numerical comparison - see module docstring).
    """
    weight, packed = _packed_weight(n_in=shape[-1])
    input = torch.randn(shape, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight.shape[0], device=flag_gems.device)

    _assert_delegates(input, packed, bias, n_out=weight.shape[0])


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
def test_fbgemm_linear_fp16_weight_fp32_activation_wrapper_matches_aten_bitwise():
    """The wrapper result is bitwise identical to calling the same aten CPU
    kernel directly.

    Both sides execute the same fbgemm CPU kernel in the same process, so this
    parity check holds on every build regardless of the fbgemm accumulation
    order (no fbgemm-free reference involved). The aten reference call is
    made with CPU-resident activations on purpose: a direct aten call with
    CUDA-resident activations makes the native kernel dereference device
    pointers and segfault (that is exactly what the wrapper's host staging
    exists to avoid), so the comparison stages the reference side the same
    way and asserts the staged result is bitwise equal.
    """
    weight, packed = _packed_weight(seed=2)
    input = torch.randn(64, N_IN, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight.shape[0], device=flag_gems.device)

    res_out = flag_gems.fbgemm_linear_fp16_weight_fp32_activation(input, packed, bias)
    aten_out = torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation(
        utils.to_reference(input).cpu(), packed, utils.to_reference(bias).cpu()
    )
    # The wrapper returns on the input device (host staging round-trips), the
    # direct CPU aten call stays on CPU; move the CPU result to the input
    # device so the shared bitwise helper compares values only.
    assert res_out.device.type == input.device.type
    torch.testing.assert_close(
        res_out.cpu(),
        aten_out.cpu(),
        atol=0,
        rtol=0,
    )


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
def test_fbgemm_linear_fp16_weight_fp32_activation_fp16_bias_accepted():
    """A fp16 bias is accepted (upcast inside the kernel) and the call routes
    through the wrapper on every build.

    No numerical comparison - see module docstring.
    """
    weight, packed = _packed_weight(seed=1)
    input = torch.randn(8, N_IN, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight.shape[0], device=flag_gems.device, dtype=torch.float16)

    _assert_delegates(input, packed, bias, n_out=weight.shape[0])


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
def test_fbgemm_linear_fp16_weight_fp32_activation_bf16_bias_accepted():
    """A bf16 bias is accepted (upcast inside the kernel)."""
    weight, packed = _packed_weight(seed=3)
    input = torch.randn(8, N_IN, device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn(weight.shape[0], device=flag_gems.device, dtype=torch.bfloat16)

    _assert_delegates(input, packed, bias, n_out=weight.shape[0])


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
def test_fbgemm_linear_fp16_weight_fp32_activation_unsupported_bias_dtype():
    """A non-fp32/fp16/bf16 bias is rejected before dispatch."""
    weight, packed = _packed_weight()
    input = torch.randn(4, N_IN, device=flag_gems.device, dtype=torch.float32)
    for bias_dtype in (torch.float64, torch.int32):
        bias = torch.randn(weight.shape[0], device=flag_gems.device).to(bias_dtype)
        with pytest.raises(RuntimeError, match="bias must have dtype"):
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
    """A strided activation is accepted and matches the contiguous result.

    Both sides go through the same wrapper, so this stays a bitwise check
    against the delegated kernel itself (no cross-implementation reference).
    """
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
