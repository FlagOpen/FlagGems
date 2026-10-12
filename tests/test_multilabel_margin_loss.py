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
from . import conftest as cfg

_ATEN_OP = torch.ops.aten.multilabel_margin_loss

if cfg.QUICK_MODE:
    # One rank<=1 shape for the 0-dim result path and one batched shape whose
    # class count spans several class tiles.
    MML_SHAPES = [(256,), (32, 257)]
    # QUICK_MODE keeps one dtype; float32 is the one whose accumulator path is
    # shared by fp16 and bf16, so it covers the most code per case.
    MML_DTYPES = [torch.float32]
else:
    # The class axis is tiled in blocks of 64 and reduction 0 takes a
    # single-launch path below 4096 classes, so the counts straddle both
    # boundaries (255/256/257 around one tile, 4096 at the launch cutoff).
    # Rank 0 and rank 1 check the 0-dim result, and the tall/narrow and
    # short/wide pairs check that few-row shapes still fill the device.
    MML_SHAPES = [
        (),
        (1,),
        (256,),
        (4096,),
        (8, 128),
        (32, 255),
        (32, 256),
        (32, 257),
        (6400, 15),
        (1024, 1024),
        (1, 4096),
        (2, 4097),
    ]
    MML_DTYPES = utils.ALL_FLOAT_DTYPES

# Reduction 0 returns per-row losses, 1 the mean over rows and 2 the sum; the
# native kernel maps every other integer to "sum", which is checked separately.
MML_REDUCTIONS = [0, 1, 2]

# Each pattern shapes the active target prefix of a row, which is what drives
# both the vector path (a single active id) and the outer-product path, and
# "full" collapses the non-target set so no pair survives at all.
MML_TARGET_PATTERNS = [
    "single",
    "half",
    "full",
    "empty",
    "ragged",
    "permuted",
    "near_full",
    "payload",
]


def _rows_cols(shape):
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        return 1, shape[0]
    return shape[0], shape[1]


def _make_target(shape, pattern="ragged", device=None):
    """Build an int64 target whose rows hold class ids ended by the first -1."""
    device = device or flag_gems.device
    rows, n_classes = _rows_cols(shape)
    out = torch.full((rows, n_classes), -1, dtype=torch.int64)
    for row in range(rows):
        ids = torch.arange(n_classes, dtype=torch.int64)
        if pattern == "permuted":
            ids = (ids * 7 + row) % n_classes
        kind = pattern
        if pattern == "ragged":
            kind = ("single", "half", "empty", "near_full")[row % 4]
        if kind == "single":
            keep = 1
        elif kind == "half":
            keep = max(1, n_classes // 2)
        elif kind == "full":
            keep = n_classes
        elif kind == "empty":
            keep = 0
        elif kind == "near_full":
            keep = max(1, n_classes - 1)
        elif kind == "payload":
            # Entries after the first -1 must be ignored, so the tail is filled
            # with ids that would change the loss if they were read.
            keep = max(1, n_classes // 3)
        else:
            keep = max(1, n_classes // 2)
        out[row, :keep] = ids[:keep]
        if kind == "payload" and keep < n_classes:
            out[row, keep] = -1
            out[row, keep + 1 :] = ids[keep + 1 :]
    return out.reshape(shape).to(device)


def _relayout(t, layout):
    """Return a view of ``t``'s values with the requested memory layout."""
    if layout == "contiguous":
        return t
    if layout == "transposed":
        if t.dim() != 2:
            return t
        buf = torch.empty(
            (t.shape[1], t.shape[0]), dtype=t.dtype, device=t.device
        ).transpose(0, 1)
        buf.copy_(t)
        return buf
    if layout == "row_strided":
        rows, cols = _rows_cols(tuple(t.shape))
        buf = torch.empty(
            (rows * 2,) + tuple(t.shape)[1:], dtype=t.dtype, device=t.device
        )
        view = buf[::2]
        view.copy_(t.reshape(view.shape))
        return view.reshape(t.shape) if t.dim() < 2 else view
    if layout == "col_strided":
        shape = tuple(t.shape)
        wide = shape[:-1] + (shape[-1] * 2,)
        buf = torch.empty(wide, dtype=t.dtype, device=t.device)
        view = buf[..., ::2]
        view.copy_(t)
        return view
    if layout == "offset":
        flat = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)
        view = flat[1:].reshape(t.shape)
        view.copy_(t)
        return view
    raise ValueError(f"unknown layout: {layout}")


def _reduce_dim(shape, reduction):
    rows, n_classes = _rows_cols(shape)
    # Each output element is a sum over up to rows * n_classes hinge terms, so
    # the comparison budget has to grow with the number of accumulated pairs.
    return n_classes if reduction == 0 else max(1, rows * n_classes)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("shape", MML_SHAPES)
@pytest.mark.parametrize("reduction", MML_REDUCTIONS)
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss(shape, reduction, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = _make_target(shape)
    ref_inp = utils.to_reference(inp)
    ref_target = utils.to_reference(target)

    ref_out = _ATEN_OP(ref_inp, ref_target, reduction)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)

    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(
        res_out, ref_out, dtype, reduce_dim=_reduce_dim(shape, reduction)
    )


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("pattern", MML_TARGET_PATTERNS)
@pytest.mark.parametrize("reduction", MML_REDUCTIONS)
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss_target_patterns(pattern, reduction, dtype):
    # 33 classes leaves a partial class tile, so the pattern's effect on the
    # active prefix is seen by both a full and a masked tile.
    shape = (6, 33)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = _make_target(shape, pattern)
    ref_inp = utils.to_reference(inp)
    ref_target = utils.to_reference(target)

    ref_out = _ATEN_OP(ref_inp, ref_target, reduction)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)
    utils.gems_assert_close(
        res_out, ref_out, dtype, reduce_dim=_reduce_dim(shape, reduction)
    )


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("n_classes", [255, 256, 257])
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss_class_count(n_classes, dtype):
    # The counts sit either side of a whole number of 64-wide class tiles, so
    # the partial-tile mask and the per-row fold are both exercised.
    shape = (4, n_classes)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = _make_target(shape)
    ref_out = _ATEN_OP(utils.to_reference(inp), utils.to_reference(target), 0)
    res_out = flag_gems.multilabel_margin_loss(inp, target, 0)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n_classes)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize(
    "in_layout", ["transposed", "row_strided", "col_strided", "offset"]
)
@pytest.mark.parametrize("tgt_layout", ["contiguous", "transposed", "col_strided"])
@pytest.mark.parametrize("reduction", MML_REDUCTIONS)
def test_multilabel_margin_loss_non_contiguous(in_layout, tgt_layout, reduction):
    # Both operands are addressed through explicit row and class strides, so a
    # view whose storage order differs from its logical order must still be read
    # in logical order.
    shape = (5, 37)
    dtype = torch.float32
    inp = _relayout(torch.randn(shape, dtype=dtype, device=flag_gems.device), in_layout)
    target = _relayout(_make_target(shape), tgt_layout)
    assert not inp.is_contiguous()

    ref_out = _ATEN_OP(utils.to_reference(inp), utils.to_reference(target), reduction)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)
    utils.gems_assert_close(
        res_out, ref_out, dtype, reduce_dim=_reduce_dim(shape, reduction)
    )


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("shape", [(32, 47), (256, 8)])
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss_default_reduction(shape, dtype):
    # The default reduction is "mean" (1), which has to produce a 0-dim result.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = _make_target(shape)
    ref_out = _ATEN_OP(utils.to_reference(inp), utils.to_reference(target), 1)
    res_out = flag_gems.multilabel_margin_loss(inp, target)
    assert res_out.shape == ()
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=_reduce_dim(shape, 1))


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("reduction", [-1, 3])
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss_extra_reduction_values(reduction, dtype):
    # Every integer other than 0 and 1 is treated as "sum" by the native kernel.
    shape = (7, 19)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = _make_target(shape)
    ref_out = _ATEN_OP(utils.to_reference(inp), utils.to_reference(target), 2)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=_reduce_dim(shape, 2))


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("reduction", MML_REDUCTIONS)
@pytest.mark.parametrize("out_layout", ["dense", "offset"])
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss_out(reduction, out_layout, dtype):
    # The .out overload resizes the caller's buffer to the result shape, writes
    # it in place and returns that same tensor, so identity matters too.
    shape = (9, 23)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = _make_target(shape)
    ref_out = _ATEN_OP(utils.to_reference(inp), utils.to_reference(target), reduction)

    if out_layout == "dense":
        out = torch.empty((), dtype=dtype, device=flag_gems.device)
    else:
        out = torch.empty(shape[0] + 1, dtype=dtype, device=flag_gems.device)[1:]
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction, out=out)
    assert res_out is out
    utils.gems_assert_close(
        res_out, ref_out, dtype, reduce_dim=_reduce_dim(shape, reduction)
    )


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("special", ["nan", "inf", "mixed"])
@pytest.mark.parametrize("layout", ["target", "non_target", "batched"])
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss_special_values(special, layout, dtype):
    # A non-finite value reached through a target id and one reached only as a
    # non-target class take different routes into the hinge, and the batched
    # case checks that one poisoned row does not leak into the others.
    shape = (3, 6) if layout == "batched" else (6,)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    bad = {"nan": float("nan"), "inf": float("inf"), "mixed": float("-inf")}[special]
    flat = inp.reshape(-1, shape[-1])
    if layout == "non_target":
        flat[:, -1] = bad
    else:
        flat[:, 0] = bad
    if special == "mixed":
        flat[:, 1] = float("nan")
    target = _make_target(shape, "half")

    ref_out = _ATEN_OP(utils.to_reference(inp), utils.to_reference(target), 0)
    res_out = flag_gems.multilabel_margin_loss(inp, target, 0)
    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        equal_nan=True,
        reduce_dim=_reduce_dim(shape, 0),
    )


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("dtype", MML_DTYPES)
def test_multilabel_margin_loss_empty_batch(dtype):
    inp = torch.empty((0, 11), dtype=dtype, device=flag_gems.device)
    target = torch.empty((0, 11), dtype=torch.int64, device=flag_gems.device)
    res_out = flag_gems.multilabel_margin_loss(inp, target, 0)
    assert res_out.shape == (0,)
    assert res_out.dtype == dtype


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("reduction", MML_REDUCTIONS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_multilabel_margin_loss_backward(reduction, dtype):
    # 29 classes leaves a partial tile in both backward kernels, and a small
    # batch keeps the atomic accumulation deterministic enough to compare.
    shape = (6, 29)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = _make_target(shape)
    ref_inp = utils.to_reference(inp).detach().requires_grad_()
    ref_target = utils.to_reference(target)

    ref_out = _ATEN_OP(ref_inp, ref_target, reduction)
    res_out = flag_gems.multilabel_margin_loss(inp, target, reduction)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad)
    (ref_dx,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_dx,) = torch.autograd.grad(res_out, inp, out_grad)
    utils.gems_assert_close(res_dx, ref_dx, dtype, reduce_dim=shape[1])


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize(
    "dtype", [torch.int8, torch.uint8, torch.int32, torch.int64, torch.float8_e4m3fn]
)
def test_multilabel_margin_loss_rejects_input_dtype(dtype):
    inp = torch.zeros((4, 8), device=flag_gems.device).to(dtype)
    target = _make_target((4, 8))
    with pytest.raises(RuntimeError):
        flag_gems.multilabel_margin_loss(inp, target, 0)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("target_dtype", [torch.int16, torch.int32, torch.float32])
def test_multilabel_margin_loss_rejects_target_dtype(target_dtype):
    inp = torch.randn((4, 8), device=flag_gems.device)
    target = _make_target((4, 8)).to(target_dtype)
    with pytest.raises(RuntimeError):
        flag_gems.multilabel_margin_loss(inp, target, 0)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("target_shape", [(4,), (3, 8), (4, 9)])
def test_multilabel_margin_loss_rejects_target_shape(target_shape):
    inp = torch.randn((4, 8), device=flag_gems.device)
    target = torch.zeros(target_shape, dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.multilabel_margin_loss(inp, target, 0)


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("reduction", ["mean", 1.5, None])
def test_multilabel_margin_loss_rejects_reduction(reduction):
    inp = torch.randn((4, 8), device=flag_gems.device)
    target = _make_target((4, 8))
    with pytest.raises(RuntimeError):
        flag_gems.multilabel_margin_loss(inp, target, reduction)


@pytest.mark.multilabel_margin_loss
def test_multilabel_margin_loss_rejects_invalid_rank():
    inp = torch.randn((2, 3, 4), device=flag_gems.device)
    target = torch.zeros((2, 3, 4), dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.multilabel_margin_loss(inp, target, 0)
