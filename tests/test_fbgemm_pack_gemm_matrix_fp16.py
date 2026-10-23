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
"""Tests for fbgemm_pack_gemm_matrix_fp16 (delegated implementation).

Reference strategy
------------------
``fbgemm_pack_gemm_matrix_fp16`` is delegated to aten (there is no Triton
kernel). It packs a fp32 weight matrix into the fbgemm GEMM fp16 format and
returns an **opaque** packed tensor: the visible ``torch.uint8`` payload
carries an fbgemm ``LinearPackedParamsBase`` C++ wrapper object
(``PackedGemmMatrixFP16``) in its TensorImpl, and the packed layout is owned
by the fbgemm library. The result is only consumable by fbgemm kernels
(``fbgemm_linear_fp16_weight`` / ``..._fp32_activation``); it has no
user-facing numerical semantics.

**Deliberately no numerical comparison of the packed output.** There is no
reference to compare against:

* the packed layout is a private fbgemm format (SIMD-blocking of the fp16
  weight), not specified by any public contract - any Python-side
  reconstruction of its bytes would mirror one build's fbgemm version, not
  the op semantics;
* the sibling tests for the fbgemm linear family (PRs #7632, #7636, #7638,
  #7647) already established that fbgemm-internal behavior differs across
  torch builds, so byte/numerical assertions on fbgemm output do not
  transfer between the local build and CI.

The tests therefore validate behavior that is identical on every build:

* the wrapper's validation logic (dtype / dim error paths), which runs
  before dispatch;
* the delegation path on a legal call - the returned tensor is an opaque
  CPU (or caller-device) uint8 tensor, which is asserted structurally
  (dtype/device) and, when the fbgemm helpers needed to *consume* a packed
  object are available, by round-tripping through
  ``fbgemm_linear_fp16_weight`` against its documented
  ``F.linear(x, w.half().float(), b)`` semantics.

Availability probing is lazy (autouse fixture via ``hasattr``), because the
fbgemm-backed helpers are deprecated on recent builds and their availability
varies across torch builds; tests that would build or consume real packed
objects skip when the environment cannot run them, so CI never depends on
fbgemm build differences.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# The pack kernel accepts fp32 weights only (fp16/fp64/bf16 are rejected)
# and returns an opaque uint8 tensor, so the dtype matrix is fixed instead of
# parametrizing over FLOAT_DTYPES.
SHAPES = (
    [(16, 32), (64, 128)]
    if utils.QUICK_MODE
    else [
        (8, 16),
        (64, 128),
        (512, 512),
    ]
)


@pytest.fixture(autouse=True)
def _require_fbgemm_pack_helper():
    """The delegation path needs the fbgemm-backed aten pack kernel.

    Probed lazily (not at import time): the op is deprecated on recent
    builds and absent on some torch builds. Validation-path tests run
    regardless; smoke/round-trip tests skip when the helper is missing.
    """
    if not hasattr(torch.ops.aten, "fbgemm_pack_gemm_matrix_fp16"):
        pytest.skip(
            "fbgemm-backed aten helper (fbgemm_pack_gemm_matrix_fp16) "
            "unavailable on this torch build"
        )


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
@pytest.mark.parametrize("shape", SHAPES)
def test_fbgemm_pack_gemm_matrix_fp16_delegates(shape):
    """A legal call delegates to the aten fbgemm kernel without raising.

    The result is the opaque packed tensor: dtype uint8, flat, on the
    input's device. No structural/numerical assertion beyond that - the
    packed layout is fbgemm-private (see module docstring).
    """
    weight = torch.randn(shape, device=flag_gems.device, dtype=torch.float32)

    packed = flag_gems.fbgemm_pack_gemm_matrix_fp16(weight)

    assert packed.dtype == torch.uint8
    assert packed.device.type == weight.device.type


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
def test_fbgemm_pack_gemm_matrix_fp16_roundtrip():
    """The packed output is consumable by its sibling aten linear kernel.

    Skipped unless both fbgemm-backed helpers exist. The round-trip is the
    documented contract of the packed object (unpack == ``w.half()``
    round-trip), so the linear result must equal ``F.linear`` on the
    fp16-rounded weight - the comparison runs through the aten CPU kernel on
    both sides, never through a fbgemm-free reference.
    """
    if not (
        hasattr(torch.ops.aten, "fbgemm_pack_gemm_matrix_fp16")
        and hasattr(torch.ops.aten, "fbgemm_linear_fp16_weight")
    ):
        pytest.skip(
            "fbgemm-backed aten helpers (fbgemm_pack_gemm_matrix_fp16 / "
            "fbgemm_linear_fp16_weight) unavailable on this torch build"
        )
    weight = torch.randn(16, 32, dtype=torch.float32)
    input = torch.randn(8, 32, dtype=torch.float32)
    bias = torch.randn(16, dtype=torch.float32)

    packed = flag_gems.fbgemm_pack_gemm_matrix_fp16(weight)
    out = torch.ops.aten.fbgemm_linear_fp16_weight(input.cpu(), packed, bias.cpu())

    ref = torch.nn.functional.linear(input, weight.half().float(), bias)
    utils.gems_assert_close(
        utils.to_reference(out), utils.to_reference(ref), dtype=torch.float32
    )


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.bfloat16])
def test_fbgemm_pack_gemm_matrix_fp16_unsupported_weight_dtype(dtype):
    """Non-fp32 weights are rejected before dispatch, like aten."""
    weight = torch.randn(4, 8, device=flag_gems.device, dtype=dtype)
    with pytest.raises(RuntimeError, match="expected scalar type Float"):
        flag_gems.fbgemm_pack_gemm_matrix_fp16(weight)


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
@pytest.mark.parametrize("shape", [(8,), (8, 16, 32), ()])
def test_fbgemm_pack_gemm_matrix_fp16_rejects_non_2d_input(shape):
    """Only matrices are accepted; other ranks are rejected like aten."""
    weight = (
        torch.randn(shape, device=flag_gems.device, dtype=torch.float32)
        if shape
        else torch.randn((), device=flag_gems.device, dtype=torch.float32)
    )
    with pytest.raises(IndexError, match="matrices only"):
        flag_gems.fbgemm_pack_gemm_matrix_fp16(weight)


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
def test_fbgemm_pack_gemm_matrix_fp16_non_contiguous_input():
    """A strided weight is accepted (aten makes it contiguous internally)."""
    weight = torch.randn(16, 32, device=flag_gems.device, dtype=torch.float32)
    strided = weight.t().t()

    packed = flag_gems.fbgemm_pack_gemm_matrix_fp16(strided)

    assert packed.dtype == torch.uint8
    assert packed.device.type == weight.device.type


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
def test_fbgemm_pack_gemm_matrix_fp16_cuda_stages_through_host():
    """CUDA input routes through the wrapper and returns on the caller device.

    The aten pack kernel has no CUDA registration and segfaults on
    CUDA-resident weights; the wrapper stages the weight through host
    memory. The returned packed tensor lands on the caller's device as a
    plain uint8 buffer (carrying the fbgemm C++ wrapper object to the GPU is
    impossible - moving any packed tensor drops the wrapper).
    """
    if flag_gems.device != "cuda":
        pytest.skip("CUDA-only routing check")
    weight = torch.randn(16, 32, device=flag_gems.device, dtype=torch.float32)

    packed = flag_gems.fbgemm_pack_gemm_matrix_fp16(weight)

    assert packed.dtype == torch.uint8
    assert packed.device.type == "cuda"
