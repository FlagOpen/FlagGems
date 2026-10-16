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

import importlib

import pytest

import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device

QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

_QINT_RANGE = {
    torch.quint8: (0, 255),
    torch.qint8: (-128, 127),
    torch.qint32: (-(2**31), 2**31 - 1),
}

QUANT_MAX_POOL2D_CONFIGS = [
    # Classic 2x2 pooling with default (empty) stride -> falls back to kernel_size
    ((2, 4, 16, 16), (2, 2), [], 0, 1, False),
    # 3x3 kernel, stride 2, padding 1 (ResNet style)
    ((4, 8, 32, 32), 3, 2, 1, 1, False),
    # Non-square kernel/stride/padding
    ((2, 3, 28, 28), (3, 5), (1, 2), (1, 0), 1, False),
    # Dilation
    ((1, 4, 20, 20), 2, 1, 0, 2, False),
    # ceil_mode
    ((2, 4, 15, 15), 3, 2, 1, 1, True),
    # Larger spatial dims
    ((1, 16, 56, 56), 3, 2, 1, 1, False),
    # Asymmetric padding
    ((2, 2, 16, 20), 2, 2, (1, 0), 1, False),
    # 3-D (C, H, W) input: the native operator treats it as a single batch
    ((3, 16, 16), (2, 2), [], 0, 1, False),
    ((5, 15, 17), 3, 2, 1, 1, True),
    # Window entirely inside the padding region (k // 2 == p), so the output
    # picks up the padding sentinel rather than a real value
    ((1, 2, 6, 6), 4, 4, 2, 1, False),
]

def _make_quant_tensor(shape, dtype, scale, zero_point, dev=device):
    """Build a per-tensor quantized tensor whose integers span the dtype range.

    The integer representation is constructed directly rather than by quantizing
    floats, so the test controls the exact ``int_repr`` (including the extremes
    of each dtype) instead of relying on the rounding of a float round-trip.
    """
    torch.manual_seed(0)
    low, high = _QINT_RANGE[dtype]
    int_dtype = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
    }[dtype]
    ints = torch.randint(low, high, shape, dtype=torch.int64).to(int_dtype)
    flat = ints.reshape(-1)
    if flat.numel() >= 2:
        # Pin the dtype extremes into the data so saturation is always covered.
        flat[0] = low
        flat[1] = high
    return torch._make_per_tensor_quantized_tensor(
        ints.to(dev), scale=scale, zero_point=zero_point
    )

def _valid_zero_point(dtype, zero_point):
    """Clamp a zero_point into the range the dtype accepts."""
    low, high = _QINT_RANGE[dtype]
    return max(low, min(high, zero_point))

def _to_cpu_keep_layout(inp):
    """Copy a quantized tensor to CPU without losing its memory format.

    ``Tensor.to("cpu")`` on a quantized tensor normalises the result to
    contiguous strides, which would silently turn a channels-last reference into
    a contiguous one.
    """
    host = inp.cpu()
    if inp.dim() == 4 and inp.is_contiguous(memory_format=torch.channels_last):
        host = host.contiguous(memory_format=torch.channels_last)
    return host

def _reference(inp, *args, **kwargs):
    """Run the native operator on a CPU copy of ``inp``."""
    return torch.quantized_max_pool2d(_to_cpu_keep_layout(inp), *args, **kwargs)

def _assert_same_quantized(res, ref, dtype):
    assert res.dtype == dtype
    assert res.shape == ref.shape
    assert res.q_scale() == ref.q_scale()
    assert res.q_zero_point() == ref.q_zero_point()
    # The integer representation is the authoritative result: max pooling picks
    # an existing quantized value, so it must be bit-exact.
    utils.gems_assert_equal(res.int_repr().cpu(), ref.int_repr())
    # Dequantize both sides on CPU. For qint32, ``dequantize`` evaluates
    # ``q - zero_point`` in int32 and wraps for integers near the dtype's
    # extremes, and CPU and CUDA wrap differently; going through CPU for both
    # compares the pooling result rather than that backend difference.
    utils.gems_assert_close(
        res.cpu().dequantize(), utils.to_reference(ref.dequantize(), True), torch.float32
    )

@pytest.mark.quantized_max_pool2d_out
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    QUANT_MAX_POOL2D_CONFIGS,
)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", [0.1, 1.5])
@pytest.mark.parametrize("zero_point", [0, -128])
def test_quantized_max_pool2d_out(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype, scale, zero_point
):
    zero_point = _valid_zero_point(dtype, zero_point)
    res_inp = _make_quant_tensor(shape, dtype, scale, zero_point)

    ref_out = _reference(
        res_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    # A correctly shaped out carrying *different* quantization parameters: the
    # operator must overwrite them with the input's, as ATen's copy_ does.
    out_tensor = torch._empty_affine_quantized(
        ref_out.shape,
        scale=scale * 3.0,
        zero_point=_valid_zero_point(dtype, zero_point + 1),
        dtype=dtype,
        device=device,
    )
    res_r = flag_gems.quantized_max_pool2d_out(
        res_inp,
        _as_int_pair(kernel_size),
        _as_int_pair(stride, allow_empty=True),
        _as_int_pair(padding),
        _as_int_pair(dilation),
        ceil_mode,
        out=out_tensor,
    )

    assert res_r is out_tensor
    assert res_r.data_ptr() == out_tensor.data_ptr()
    _assert_same_quantized(res_r, ref_out, dtype)

def _as_int_pair(value, allow_empty=False):
    """Expand a pooling parameter to the 2-element list the schema wants."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value, value]
    value = list(value)
    if not value and allow_empty:
        return []
    return value

@pytest.mark.quantized_max_pool2d_out
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantized_max_pool2d_out_resizes(dtype):
    """A wrongly shaped out is resized to the result's shape, as ATen does."""
    res_inp = _make_quant_tensor(
        (2, 3, 16, 16), dtype, 0.25, _valid_zero_point(dtype, 4)
    )
    ref_out = _reference(res_inp, (2, 2))

    for out_shape in [(1,), (0,), (2, 3, 2, 2), (2, 3, 9, 9), (7,)]:
        out_tensor = torch._empty_affine_quantized(
            out_shape, scale=9.0, zero_point=0, dtype=dtype, device=device
        )
        res_r = flag_gems.quantized_max_pool2d_out(
            res_inp, [2, 2], [], [0, 0], [1, 1], False, out=out_tensor
        )
        assert res_r is out_tensor
        assert tuple(res_r.shape) == tuple(ref_out.shape)
        _assert_same_quantized(res_r, ref_out, dtype)

@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out_preserves_offset_and_strides():
    """A strided/offset out is written through its own strides.

    ``out`` is a non-contiguous view into a larger buffer; the operator must fill
    exactly that view and leave the surrounding elements untouched.
    """
    res_inp = _make_quant_tensor((2, 3, 8, 8), torch.quint8, 0.25, 7)
    ref_out = _reference(res_inp, (2, 2))

    # Strided view: every other column of a wider buffer.
    buffer = torch._empty_affine_quantized(
        (2, 3, 4, 8), scale=0.25, zero_point=7, dtype=torch.quint8, device=device
    )
    buffer.copy_(
        torch._make_per_tensor_quantized_tensor(
            torch.full((2, 3, 4, 8), 11, dtype=torch.uint8, device=device),
            scale=0.25,
            zero_point=7,
        )
    )
    view = buffer[..., ::2]
    pointer_before = view.data_ptr()
    stride_before = view.stride()
    offset_before = view.storage_offset()
    result = flag_gems.quantized_max_pool2d_out(
        res_inp, [2, 2], [], [0, 0], [1, 1], False, out=view
    )
    assert result is view
    assert result.data_ptr() == pointer_before
    assert result.stride() == stride_before
    assert result.storage_offset() == offset_before
    utils.gems_assert_equal(view.int_repr().cpu(), ref_out.int_repr())
    # The interleaved columns that are not part of ``out`` keep their sentinel.
    assert bool((buffer[..., 1::2].int_repr() == 11).all().item())

    # Offset view: the second slice of a batched buffer.
    big = torch._empty_affine_quantized(
        (2, 2, 3, 4, 4), scale=0.25, zero_point=7, dtype=torch.quint8, device=device
    )
    big.copy_(
        torch._make_per_tensor_quantized_tensor(
            torch.full((2, 2, 3, 4, 4), 11, dtype=torch.uint8, device=device),
            scale=0.25,
            zero_point=7,
        )
    )
    offset_view = big[1]
    pointer_before = offset_view.data_ptr()
    stride_before = offset_view.stride()
    offset_before = offset_view.storage_offset()
    assert offset_before != 0
    result = flag_gems.quantized_max_pool2d_out(
        res_inp, [2, 2], [], [0, 0], [1, 1], False, out=offset_view
    )
    assert result is offset_view
    assert result.data_ptr() == pointer_before
    assert result.stride() == stride_before
    assert result.storage_offset() == offset_before
    utils.gems_assert_equal(offset_view.int_repr().cpu(), ref_out.int_repr())
    assert bool((big[0].int_repr() == 11).all().item())

@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out_channels_last_out():
    """A channels_last out of the right shape keeps its layout."""
    res_inp = _make_quant_tensor((2, 3, 8, 8), torch.quint8, 0.25, 7)
    ref_out = _reference(res_inp, (2, 2))

    out_tensor = torch._empty_affine_quantized(
        (2, 3, 4, 4),
        scale=0.25,
        zero_point=7,
        dtype=torch.quint8,
        device=device,
        memory_format=torch.channels_last,
    )
    flag_gems.quantized_max_pool2d_out(
        res_inp, [2, 2], [], [0, 0], [1, 1], False, out=out_tensor
    )
    assert out_tensor.is_contiguous(memory_format=torch.channels_last)
    utils.gems_assert_equal(out_tensor.int_repr().cpu(), ref_out.int_repr())

@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out_aliasing_input():
    """An out that aliases the input still produces the functional result."""
    res_inp = _make_quant_tensor((1, 1, 4, 4), torch.quint8, 0.5, 0)
    ref_out = _reference(res_inp, [2, 2])

    res_r = flag_gems.quantized_max_pool2d_out(
        res_inp, [2, 2], [], [0, 0], [1, 1], False, out=res_inp
    )
    utils.gems_assert_equal(res_r.int_repr().cpu(), ref_out.int_repr())

@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out_mismatched_qparams():
    """A per-tensor out with different qparams adopts the input's.

    ``out.copy_`` is what moves the quantizer in ATen; the operator must
    reproduce that without a second native pass over the values, so this pins
    the observable contract (same object, resized shape, input's qparams).
    """
    res_inp = _make_quant_tensor((2, 3, 8, 8), torch.quint8, 0.25, 7)
    ref_out = _reference(res_inp, (2, 2))

    out_tensor = torch._empty_affine_quantized(
        (1,), scale=9.0, zero_point=1, dtype=torch.quint8, device=device
    )
    res_r = flag_gems.quantized_max_pool2d_out(
        res_inp, [2, 2], [], [0, 0], [1, 1], False, out=out_tensor
    )
    assert res_r is out_tensor
    assert tuple(res_r.shape) == tuple(ref_out.shape)
    assert res_r.q_scale() == res_inp.q_scale()
    assert res_r.q_zero_point() == res_inp.q_zero_point()
    _assert_same_quantized(res_r, ref_out, torch.quint8)

@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out_rejects_per_channel_out():
    """A per-channel out raises ATen's "same qscheme" message.

    Nothing can be written in that case: the kernel stores integers and has no
    way to install a per-tensor quantizer over a per-channel tensor, which is
    why ATen's quantized ``copy_`` rejects it.
    """
    torch.manual_seed(0)
    res_inp = _make_quant_tensor((2, 3, 8, 8), torch.quint8, 0.25, 7)
    per_channel = torch.quantize_per_channel(
        torch.randn(2, 3, 4, 4, device=device),
        torch.rand(3, device=device) + 0.1,
        torch.zeros(3, dtype=torch.long, device=device),
        1,
        torch.quint8,
    )

    if device == "cpu":
        with pytest.raises(RuntimeError, match="same qscheme"):
            torch.ops.aten.quantized_max_pool2d.out(
                _to_cpu_keep_layout(res_inp),
                [2, 2],
                [],
                [0, 0],
                [1, 1],
                False,
                out=per_channel,
            )
    with pytest.raises(RuntimeError, match="same qscheme"):
        flag_gems.quantized_max_pool2d_out(
            res_inp, [2, 2], [], [0, 0], [1, 1], False, out=per_channel
        )

@pytest.mark.quantized_max_pool2d_out
def test_quantized_max_pool2d_out_rejects_mismatched_out():
    """dtype/device mismatches raise the same messages as ATen."""
    res_inp = _make_quant_tensor((2, 3, 8, 8), torch.quint8, 0.25, 7)

    bad_dtype = torch._empty_affine_quantized(
        (2, 3, 4, 4), scale=0.25, zero_point=0, dtype=torch.qint8, device=device
    )
    with pytest.raises(RuntimeError, match="dtype c10::quint8"):
        flag_gems.quantized_max_pool2d_out(
            res_inp, [2, 2], [], [0, 0], [1, 1], False, out=bad_dtype
        )

    bad_device = torch._empty_affine_quantized(
        (2, 3, 4, 4), scale=0.25, zero_point=7, dtype=torch.quint8, device="cpu"
    )
    with pytest.raises(RuntimeError, match="Expected out tensor to have device"):
        flag_gems.quantized_max_pool2d_out(
            res_inp, [2, 2], [], [0, 0], [1, 1], False, out=bad_device
        )

INVALID_PARAM_CASES = [
    (dict(kernel_size=[2, 2, 2]), "Expected 1d or 2d kernel size, got 3"),
    (dict(kernel_size=[]), "Expected 1d or 2d kernel size, got 0"),
    (
        dict(kernel_size=[2, 2], stride=[2]),
        "Expected no strides or 2d strides, got1",
    ),
    (
        dict(kernel_size=[2, 2], stride=[2, 2, 2]),
        "Expected no strides or 2d strides, got3",
    ),
    (
        dict(kernel_size=[3, 3], padding=[1, 1, 1]),
        "Expected 1d or 2d padding, got 3",
    ),
    (
        dict(kernel_size=[3, 3], dilation=[1, 1, 1]),
        "Expected 1d or 2d dilation, got 3",
    ),
    (dict(kernel_size=[2, 2], dilation=[0, 0]), "Expected dilation >= 1"),
    (dict(kernel_size=[2, 2], dilation=[-1, -1]), "Expected dilation >= 1"),
    (dict(kernel_size=[0, 0]), "kernel_size should be greater than zero."),
    (dict(kernel_size=[-1, -1]), "kernel_size should be greater than zero."),
    (
        dict(kernel_size=[2, 2], stride=[0, 0]),
        "strides should be greater than zero.",
    ),
    (
        dict(kernel_size=[2, 2], stride=[-1, -1]),
        "strides should be greater than zero.",
    ),
    (
        dict(kernel_size=[2, 2], padding=[-1, -1]),
        "pad must be non-negative, but got pad: -1",
    ),
    (
        dict(kernel_size=[2, 2], padding=[2, 2]),
        "padding should be smaller than half of kernel_size.",
    ),
    (
        dict(kernel_size=[3, 3], padding=[2, 2]),
        "padding should be smaller than half of kernel_size.",
    ),
    (
        dict(kernel_size=[16, 16]),
        "Given input size: (3x8x8). Calculated output size: (3x0x0). "
        "Output size is too small.",
    ),
]
