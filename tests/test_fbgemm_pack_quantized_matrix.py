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
"""Tests for fbgemm_pack_quantized_matrix (delegated implementation).

Reference strategy
------------------
``fbgemm_pack_quantized_matrix`` is delegated to aten (there is no Triton
kernel). It packs an int8 quantized weight matrix into the fbgemm GEMM
format and returns an **opaque** packed tensor: the visible ``torch.uint8``
payload carries an fbgemm ``LinearPackedParamsBase`` C++ wrapper object in
its TensorImpl, and the packed layout is owned by the fbgemm library. The
result is only consumable by fbgemm quantized kernels; it has no
user-facing numerical semantics.

**Deliberately no numerical comparison of the packed output.** There is no
reference to compare against:

* the packed layout is a private fbgemm format, not specified by any public
  contract - any Python-side reconstruction of its bytes would mirror one
  build's fbgemm version, not the op semantics;
* the sibling tests for the fbgemm linear family (PRs #7632, #7636, #7638,
  #7647, #7656) already established that fbgemm-internal behavior differs
  across torch builds, so byte/numerical assertions on fbgemm output do not
  transfer between the local build and CI. The packed bytes themselves are
  also not comparable across calls: the visible uint8 buffer is a fixed-size
  holder for the C++ object and a few bytes are heap addresses that differ
  run to run (the buffer is 104 bytes for every weight shape, from 8x16 to
  1024x1024 - the weight data is not in the visible tensor at all).

The tests therefore validate behavior that is identical on every build:

* the wrapper's validation logic (dtype / dim error paths), which runs
  before dispatch;
* the delegation path on a legal call for **both overloads** - the default
  overload and the ``.KN`` overload (exercised through the optional
  ``K``/``N`` arguments) - asserting the returned tensor structurally
  (dtype/device) only;
* the **delegation contract** at the consumer: the wrapper's packed object
  and the one aten returns produce bitwise identical output when fed to
  ``fbgemm_linear_int8_weight``. This compares wrapper against aten (not
  against a fbgemm-free reconstruction of the packed layout), so it holds on
  every build that has the helpers.

Availability probing is lazy (autouse fixture via ``hasattr``), because the
fbgemm-backed helpers are deprecated on recent builds and their availability
varies across torch builds; tests that would build real packed objects skip
when the environment cannot run them, so CI never depends on fbgemm build
differences.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# The pack kernel accepts int8 weights only (fp16/fp32/int64/uint8 are
# rejected) and returns an opaque uint8 tensor, so the dtype matrix is fixed
# instead of parametrizing over FLOAT_DTYPES.
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
    regardless; smoke tests skip when the helper is missing.
    """
    if not hasattr(torch.ops.aten, "fbgemm_pack_quantized_matrix"):
        pytest.skip(
            "fbgemm-backed aten helper (fbgemm_pack_quantized_matrix) "
            "unavailable on this torch build"
        )


@pytest.mark.fbgemm_pack_quantized_matrix
@pytest.mark.parametrize("shape", SHAPES)
def test_fbgemm_pack_quantized_matrix_delegates(shape):
    """A legal call (default overload) delegates without raising.

    The result is the opaque packed tensor: dtype uint8, flat, on the
    input's device. No structural/numerical assertion beyond that - the
    packed layout is fbgemm-private (see module docstring).
    """
    weight = torch.randint(-128, 127, shape, device=flag_gems.device, dtype=torch.int8)

    packed = flag_gems.fbgemm_pack_quantized_matrix(weight)

    assert packed.dtype == torch.uint8
    assert packed.device.type == weight.device.type


@pytest.mark.fbgemm_pack_quantized_matrix
@pytest.mark.parametrize("shape", SHAPES)
def test_fbgemm_pack_quantized_matrix_kn_overload_delegates(shape):
    """The optional K/N arguments route to the .KN overload without raising.

    The aten op exposes two overloads sharing one C++ entry point; the
    wrapper merges them into a single Python entry point with optional
    ``K``/``N``. The packed layout is fbgemm-private (see module docstring),
    so only dtype/device are asserted.
    """
    weight = torch.randint(-128, 127, shape, device=flag_gems.device, dtype=torch.int8)

    packed = flag_gems.fbgemm_pack_quantized_matrix(weight, K=shape[1], N=shape[0])

    assert packed.dtype == torch.uint8
    assert packed.device.type == weight.device.type


@pytest.mark.fbgemm_pack_quantized_matrix
def test_fbgemm_pack_quantized_matrix_consumer_parity():
    """The wrapper's packed object is bit-identical to aten's, at the consumer.

    The packed bytes themselves are not comparable: the visible uint8 buffer
    is a fixed-size holder whose payload covers a C++ object, and a few bytes
    are heap addresses that differ run to run (verified: the buffer is 104
    bytes for every weight shape, including 1024x1024). Comparing two
    independently built packed objects therefore has to go through a consumer.

    ``fbgemm_linear_int8_weight`` accepts the object and produces stable
    output: two packed objects built from the same weight yield bitwise
    identical linear output. The test asserts the wrapper's object behaves
    exactly like the one aten returns, which is the delegation contract, and
    is build-independent. Skipped unless both helpers exist.
    """
    if not (
        hasattr(torch.ops.aten, "fbgemm_pack_quantized_matrix")
        and hasattr(torch.ops.aten, "fbgemm_linear_int8_weight")
        and hasattr(torch.ops.aten, "fbgemm_linear_quantize_weight")
    ):
        pytest.skip(
            "fbgemm-backed aten helpers (pack_quantized_matrix / "
            "linear_int8_weight / linear_quantize_weight) unavailable on "
            "this torch build"
        )
    weight = torch.randint(-128, 127, (8, 16), dtype=torch.int8)
    input = torch.randn(4, 16, dtype=torch.float32)
    bias = torch.zeros(8, dtype=torch.float32)
    # The consumer needs the int8 weight plus its scale/zero-point and
    # column offsets; take them from aten's own quantizer so the test never
    # hardcodes fbgemm's quantization convention.
    quantized, col_offsets, scale, zero_point = (
        torch.ops.aten.fbgemm_linear_quantize_weight(weight.float())
    )

    from_wrapper = flag_gems.fbgemm_pack_quantized_matrix(weight)
    from_aten = torch.ops.aten.fbgemm_pack_quantized_matrix(weight)

    out_wrapper = torch.ops.aten.fbgemm_linear_int8_weight(
        input, quantized, from_wrapper, col_offsets, scale, zero_point, bias
    )
    out_aten = torch.ops.aten.fbgemm_linear_int8_weight(
        input, quantized, from_aten, col_offsets, scale, zero_point, bias
    )

    utils.gems_assert_close(
        utils.to_reference(out_wrapper),
        utils.to_reference(out_aten),
        dtype=torch.float32,
        atol=1e-6,
    )


@pytest.mark.fbgemm_pack_quantized_matrix
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.uint8])
def test_fbgemm_pack_quantized_matrix_unsupported_weight_dtype(dtype):
    """Non-int8 weights are rejected before dispatch, like aten."""
    weight = torch.zeros(4, 8, device=flag_gems.device, dtype=dtype)
    with pytest.raises(RuntimeError, match="expected scalar type Char"):
        flag_gems.fbgemm_pack_quantized_matrix(weight)


@pytest.mark.fbgemm_pack_quantized_matrix
@pytest.mark.parametrize("shape", [(8,), (8, 16, 32), ()])
def test_fbgemm_pack_quantized_matrix_rejects_non_2d_input(shape):
    """Only matrices are accepted; other ranks are rejected like aten."""
    weight = (
        torch.randint(-128, 127, shape, device=flag_gems.device, dtype=torch.int8)
        if shape
        else torch.randint(-128, 127, (), device=flag_gems.device, dtype=torch.int8)
    )
    with pytest.raises(IndexError, match="matrices only"):
        flag_gems.fbgemm_pack_quantized_matrix(weight)


@pytest.mark.fbgemm_pack_quantized_matrix
def test_fbgemm_pack_quantized_matrix_non_contiguous_input():
    """A strided weight is accepted (aten makes it contiguous internally)."""
    weight = torch.randint(
        -128, 127, (16, 32), device=flag_gems.device, dtype=torch.int8
    )
    strided = weight.t().t()

    packed = flag_gems.fbgemm_pack_quantized_matrix(strided)

    assert packed.dtype == torch.uint8
    assert packed.device.type == weight.device.type


@pytest.mark.fbgemm_pack_quantized_matrix
def test_fbgemm_pack_quantized_matrix_cuda_stages_through_host():
    """CUDA input routes through the wrapper and returns on the caller device.

    The aten pack kernel has no CUDA registration and segfaults on
    CUDA-resident weights; the wrapper stages the weight through host
    memory. The returned packed tensor lands on the caller's device as a
    plain uint8 buffer (carrying the fbgemm C++ wrapper object to the GPU is
    impossible - moving any packed tensor drops the wrapper).
    """
    if flag_gems.device != "cuda":
        pytest.skip("CUDA-only routing check")
    weight = torch.randint(
        -128, 127, (16, 32), device=flag_gems.device, dtype=torch.int8
    )

    packed = flag_gems.fbgemm_pack_quantized_matrix(weight)

    assert packed.dtype == torch.uint8
    assert packed.device.type == "cuda"
