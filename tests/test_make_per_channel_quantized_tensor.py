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

# _make_per_channel_quantized_tensor wraps a raw integer payload with
# per-channel quantization metadata. The output quantized dtype follows the
# payload's *integer* dtype (int8 -> qint8, uint8 -> quint8), so the test
# parametrizes over payload dtypes instead of utils.FLOAT_DTYPES. The
# QuantizedCUDA backend has no kernel for int32 storage
# (_empty_per_channel_affine_quantized raises NotImplementedError), matching
# ATen, so qint32 is excluded from the device matrix.
PAYLOAD_DTYPES = [torch.int8, torch.uint8]
QDTYPE_FROM_PAYLOAD = {torch.int8: torch.qint8, torch.uint8: torch.quint8}

# Measured ATen contract: the scale/zero-point vectors are *not* validated
# against self.size(axis) by this op (unlike quantize_per_channel) -- any 1-D
# equal-length pair is accepted and stored verbatim; ``axis`` is stored
# as-is (even negative). The tests use shape-matching vectors so the result
# is dequantizable and comparable through torch.dequantize().
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


def _make_inputs(shape, axis, device):
    payload_dtype = torch.int8
    payload = torch.randint(-100, 100, shape, device=device, dtype=payload_dtype)
    n_channels = shape[axis]
    scales = torch.rand(n_channels, device=device) * 0.5 + 0.01
    zero_points = torch.randint(-40, 40, (n_channels,), device=device, dtype=torch.int8)
    return payload, scales, zero_points


def _assert_quantized_identical(res, ref):
    """The wrap is a metadata pack: payload and stored params must be exact."""
    assert res.dtype == ref.dtype, f"dtype mismatch: {res.dtype} vs {ref.dtype}"
    assert res.qscheme() == ref.qscheme(), f"qscheme {res.qscheme()} vs {ref.qscheme()}"
    assert res.q_per_channel_axis() == ref.q_per_channel_axis(), "axis mismatch"
    utils.gems_assert_equal(res.q_per_channel_scales(), ref.q_per_channel_scales())
    res_zp = res.q_per_channel_zero_points()
    ref_zp = ref.q_per_channel_zero_points()
    assert res_zp.dtype == ref_zp.dtype, f"zp dtype {res_zp.dtype} vs {ref_zp.dtype}"
    utils.gems_assert_equal(res_zp, ref_zp)
    utils.gems_assert_equal(res.int_repr(), ref.int_repr())


def _reference(payload, scales, zero_points, axis, device_ref):
    payload_r = utils.to_reference(payload)
    scales_r = utils.to_reference(scales)
    zero_points_r = utils.to_reference(zero_points)
    return torch.ops.aten._make_per_channel_quantized_tensor(
        payload_r, scales_r, zero_points_r, axis
    )


@pytest.mark.make_per_channel_quantized_tensor
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_make_per_channel_quantized_tensor(shape, axis):
    if axis >= len(shape):
        pytest.skip(f"axis {axis} out of range for {len(shape)}-d shape")
    payload, scales, zero_points = _make_inputs(shape, axis, flag_gems.device)

    ref_out = _reference(payload, scales, zero_points, axis, flag_gems.device)
    res_out = flag_gems._make_per_channel_quantized_tensor(
        payload, scales, zero_points, axis
    )

    _assert_quantized_identical(res_out, ref_out)
    # dequantize exercises the (scale, zero_point, axis) triple end to end.
    # A negative axis is stored verbatim by both implementations and makes
    # dequantize raise "Channel axis out of range" on both paths, so only the
    # non-negative axes get the dequantize check.
    if axis >= 0:
        utils.gems_assert_close(
            torch.dequantize(res_out),
            torch.dequantize(ref_out),
            dtype=torch.float32,
        )


@pytest.mark.make_per_channel_quantized_tensor
@pytest.mark.parametrize("shape", SHAPES)
def test_make_per_channel_quantized_tensor_last_axis(shape):
    """Exercise the last (innermost) channel axis specifically."""
    axis = len(shape) - 1
    payload, scales, zero_points = _make_inputs(shape, axis, flag_gems.device)

    ref_out = _reference(payload, scales, zero_points, axis, flag_gems.device)
    res_out = flag_gems._make_per_channel_quantized_tensor(
        payload, scales, zero_points, axis
    )

    _assert_quantized_identical(res_out, ref_out)


@pytest.mark.make_per_channel_quantized_tensor
@pytest.mark.parametrize("shape", SHAPES)
def test_make_per_channel_quantized_tensor_quint8(shape):
    """uint8 payload maps to quint8 and zero-extends into int64 storage."""
    axis = 0
    payload = torch.randint(0, 255, shape, device=flag_gems.device, dtype=torch.uint8)
    n_channels = shape[axis]
    scales = torch.rand(n_channels, device=flag_gems.device) * 0.5 + 0.01
    zero_points = torch.randint(
        0, 200, (n_channels,), device=flag_gems.device, dtype=torch.uint8
    )

    ref_out = _reference(payload, scales, zero_points, axis, flag_gems.device)
    res_out = flag_gems._make_per_channel_quantized_tensor(
        payload, scales, zero_points, axis
    )

    assert res_out.dtype == torch.quint8
    _assert_quantized_identical(res_out, ref_out)


@pytest.mark.make_per_channel_quantized_tensor
@pytest.mark.parametrize("shape", [(32, 64), (16, 32, 8)])
def test_make_per_channel_quantized_tensor_float_zparams(shape):
    """A floating-point zero_point selects per_channel_affine_float_qparams."""
    axis = 0
    payload = torch.randint(-100, 100, shape, device=flag_gems.device, dtype=torch.int8)
    n_channels = shape[axis]
    scales = torch.rand(n_channels, device=flag_gems.device) * 0.5 + 0.01
    zero_points = torch.rand(n_channels, device=flag_gems.device) * 100.0

    ref_out = _reference(payload, scales, zero_points, axis, flag_gems.device)
    res_out = flag_gems._make_per_channel_quantized_tensor(
        payload, scales, zero_points, axis
    )

    assert res_out.qscheme() == torch.per_channel_affine_float_qparams
    _assert_quantized_identical(res_out, ref_out)


@pytest.mark.make_per_channel_quantized_tensor
@pytest.mark.parametrize("shape", [(16, 32)])
def test_make_per_channel_quantized_tensor_non_contiguous(shape):
    """A strided payload is gathered into fresh contiguous quantized storage."""
    axis = 0
    pool = torch.zeros(
        shape[0] * 2, shape[1], device=flag_gems.device, dtype=torch.int8
    )
    payload = pool[::2]
    payload.copy_(
        torch.randint(-100, 100, shape, device=flag_gems.device, dtype=torch.int8)
    )
    n_channels = shape[axis]
    scales = torch.rand(n_channels, device=flag_gems.device) * 0.5 + 0.01
    zero_points = torch.randint(
        -40, 40, (n_channels,), device=flag_gems.device, dtype=torch.int8
    )

    ref_out = _reference(payload, scales, zero_points, axis, flag_gems.device)
    res_out = flag_gems._make_per_channel_quantized_tensor(
        payload, scales, zero_points, axis
    )

    _assert_quantized_identical(res_out, ref_out)
    assert res_out.int_repr().is_contiguous()


@pytest.mark.make_per_channel_quantized_tensor_out
def test_make_per_channel_quantized_tensor_out():
    """The .out overload adopts the source quantizer through quantized copy_.

    The ``.out`` overload mirrors the sibling ``quantize_per_channel`` pair:
    a separate yaml id and a ``_out`` mark, so CI marker selection keeps both
    variants runnable.
    """
    # 2-D weight-shaped payload with a leading channel axis.
    shape = (16, 32)
    axis = 0
    payload, scales, zero_points = _make_inputs(shape, axis, flag_gems.device)

    ref_out = torch.ops.aten._make_per_channel_quantized_tensor(
        utils.to_reference(payload),
        utils.to_reference(scales),
        utils.to_reference(zero_points),
        axis,
    )
    res_out = flag_gems._make_per_channel_quantized_tensor(
        payload, scales, zero_points, axis
    )

    # Pre-allocate the destination through the public quantize_per_channel
    # factory on a zeros tensor with matching qparams -- the FlagGems out
    # kernel overwrites the integer storage and adopts the source quantizer
    # (quantized copy_ semantics), exactly like the sibling
    # benchmark/test_quantize_per_channel.py out= destination.
    out = torch.quantize_per_channel(
        torch.zeros(shape, dtype=torch.float32, device=flag_gems.device),
        scales.double(),
        zero_points.long(),
        axis,
        torch.qint8,
    )
    flag_gems._make_per_channel_quantized_tensor_out(
        payload, scales, zero_points, axis, out=out
    )
    # Under --ref cpu the reference moves to CPU while the FlagGems results
    # stay on the accelerator; route both sides through to_reference so the
    # shared helper's device convention holds.
    _assert_quantized_identical(utils.to_reference(out), utils.to_reference(res_out))
    _assert_quantized_identical(utils.to_reference(out), utils.to_reference(ref_out))
