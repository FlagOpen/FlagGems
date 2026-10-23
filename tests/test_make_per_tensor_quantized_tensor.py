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

import flag_gems

from . import accuracy_utils as utils

# _make_per_tensor_quantized_tensor wraps a raw integer payload with
# per-tensor quantization metadata (a single scalar scale / zero_point). The
# output quantized dtype follows the payload's *integer* dtype (int8 -> qint8,
# uint8 -> quint8), so the test parametrizes over payload dtypes instead of
# utils.FLOAT_DTYPES. The QuantizedCUDA backend has no kernel for int32
# storage (_empty_affine_quantized raises NotImplementedError), matching ATen,
# so qint32 is excluded from the device matrix.
PAYLOAD_DTYPES = [torch.int8, torch.uint8]
QDTYPE_FROM_PAYLOAD = {torch.int8: torch.qint8, torch.uint8: torch.quint8}

# Measured ATen contract: the scalar scale / zero_point are stored verbatim
# with no range check (zero_point=-129 for qint8 and 300 for quint8 are all
# accepted). The tests use in-range values so the result is dequantizable and
# comparable through torch.dequantize().
SHAPES = (
    [(8, 16), (16, 32, 4)]
    if utils.QUICK_MODE
    else [
        (8, 16),
        (64, 128),
        (1024, 1024),
        (16, 128, 64),
        (16, 7, 57, 32),
    ]
)


def _make_inputs(shape, device):
    payload = torch.randint(-100, 100, shape, device=device, dtype=torch.int8)
    return payload, 0.05, 10


def _assert_quantized_identical(res, ref):
    """The wrap is a metadata pack: payload and stored params must be exact."""
    assert res.dtype == ref.dtype, f"dtype mismatch: {res.dtype} vs {ref.dtype}"
    assert res.qscheme() == ref.qscheme(), f"qscheme {res.qscheme()} vs {ref.qscheme()}"
    assert (
        res.q_scale() == ref.q_scale()
    ), f"scale {res.q_scale()!r} vs {ref.q_scale()!r}"
    assert res.q_zero_point() == ref.q_zero_point(), "zero_point mismatch"
    utils.gems_assert_equal(res.int_repr(), ref.int_repr())


def _reference(payload, scale, zero_point, device_ref):
    payload_r = utils.to_reference(payload)
    return torch.ops.aten._make_per_tensor_quantized_tensor(
        payload_r, scale, zero_point
    )


@pytest.mark.make_per_tensor_quantized_tensor
@pytest.mark.parametrize("shape", SHAPES)
def test_make_per_tensor_quantized_tensor(shape):
    payload, scale, zero_point = _make_inputs(shape, flag_gems.device)

    ref_out = _reference(payload, scale, zero_point, flag_gems.device)
    res_out = flag_gems._make_per_tensor_quantized_tensor(payload, scale, zero_point)

    _assert_quantized_identical(res_out, ref_out)
    # dequantize exercises the (scale, zero_point) pair end to end.
    utils.gems_assert_close(
        torch.dequantize(res_out),
        torch.dequantize(ref_out),
        dtype=torch.float32,
    )


@pytest.mark.make_per_tensor_quantized_tensor
@pytest.mark.parametrize("shape", SHAPES)
def test_make_per_tensor_quantized_tensor_quint8(shape):
    """uint8 payload maps to quint8."""
    payload = torch.randint(0, 255, shape, device=flag_gems.device, dtype=torch.uint8)
    scale, zero_point = 0.1, 128

    ref_out = _reference(payload, scale, zero_point, flag_gems.device)
    res_out = flag_gems._make_per_tensor_quantized_tensor(payload, scale, zero_point)

    assert res_out.dtype == torch.quint8
    _assert_quantized_identical(res_out, ref_out)


@pytest.mark.make_per_tensor_quantized_tensor
@pytest.mark.parametrize("shape", [(32, 64)])
def test_make_per_tensor_quantized_tensor_zero_point_no_range_check(shape):
    """ATen stores the zero_point verbatim, without any range validation."""
    payload = torch.randint(-100, 100, shape, device=flag_gems.device, dtype=torch.int8)

    ref_out = _reference(payload, 0.1, 1000, flag_gems.device)
    res_out = flag_gems._make_per_tensor_quantized_tensor(payload, 0.1, 1000)

    assert res_out.q_zero_point() == 1000
    _assert_quantized_identical(res_out, ref_out)


@pytest.mark.make_per_tensor_quantized_tensor
@pytest.mark.parametrize("shape", [(16, 32)])
def test_make_per_tensor_quantized_tensor_non_contiguous(shape):
    """A strided payload is gathered into fresh contiguous quantized storage."""
    pool = torch.zeros(
        shape[0] * 2, shape[1], device=flag_gems.device, dtype=torch.int8
    )
    payload = pool[::2]
    payload.copy_(
        torch.randint(-100, 100, shape, device=flag_gems.device, dtype=torch.int8)
    )

    ref_out = _reference(payload, 0.05, 10, flag_gems.device)
    res_out = flag_gems._make_per_tensor_quantized_tensor(payload, 0.05, 10)

    _assert_quantized_identical(res_out, ref_out)
    assert res_out.int_repr().is_contiguous()


@pytest.mark.make_per_tensor_quantized_tensor
def test_make_per_tensor_quantized_tensor_scalar_payload():
    """A 0-dim payload keeps its shape through the wrap."""
    payload = torch.tensor(-5, device=flag_gems.device, dtype=torch.int8)

    ref_out = _reference(payload, 0.1, 0, flag_gems.device)
    res_out = flag_gems._make_per_tensor_quantized_tensor(payload, 0.1, 0)

    assert res_out.shape == ()
    _assert_quantized_identical(res_out, ref_out)


@pytest.mark.make_per_tensor_quantized_tensor
@pytest.mark.parametrize("dtype", [torch.int16, torch.float32])
def test_make_per_tensor_quantized_tensor_unsupported_dtype(dtype):
    """Non-storage payload dtypes hit the same backend gap as ATen."""
    payload = torch.zeros((4, 4), device=flag_gems.device, dtype=dtype)
    with pytest.raises(NotImplementedError):
        flag_gems._make_per_tensor_quantized_tensor(payload, 0.1, 0)


@pytest.mark.make_per_tensor_quantized_tensor_out
def test_make_per_tensor_quantized_tensor_out():
    """The .out overload adopts the source quantizer through quantized copy_.

    The ``.out`` overload mirrors the sibling ``quantize_per_tensor`` pair:
    a separate yaml id and a ``_out`` mark, so CI marker selection keeps both
    variants runnable.
    """
    # A single 2-D weight-like shape exercises the .out contract (same qdtype
    # destination, quantizer adoption through quantized copy_); the wrap paths
    # themselves are covered by the shape-parametrized cases above.
    shape = (16, 32)
    scale, zero_point = 0.2, 7
    payload = torch.randint(-100, 100, shape, device=flag_gems.device, dtype=torch.int8)

    ref_out = torch.ops.aten._make_per_tensor_quantized_tensor(
        utils.to_reference(payload), scale, zero_point
    )
    res_out = flag_gems._make_per_tensor_quantized_tensor(payload, scale, zero_point)

    # Pre-allocate the destination through the public quantize_per_tensor
    # factory on a zeros tensor with matching qparams -- the FlagGems out
    # kernel overwrites the integer storage and adopts the source quantizer
    # (quantized copy_ semantics), exactly like the sibling benchmark out=
    # destination.
    out = torch.quantize_per_tensor(
        torch.zeros(shape, dtype=torch.float32, device=flag_gems.device),
        0.9,
        3,
        torch.qint8,
    )
    flag_gems._make_per_tensor_quantized_tensor_out(payload, scale, zero_point, out=out)
    # Under --ref cpu the reference moves to CPU while the FlagGems results
    # stay on the accelerator; route both sides through to_reference so the
    # shared helper's device convention holds.
    _assert_quantized_identical(utils.to_reference(out), utils.to_reference(res_out))
    _assert_quantized_identical(utils.to_reference(out), utils.to_reference(ref_out))
