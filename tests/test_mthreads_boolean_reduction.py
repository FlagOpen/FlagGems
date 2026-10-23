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

"""MTHREADS boolean-reduction regression tests (all / any, *_dim, *_dims).

Pins the correctness defects previously found in the MTHREADS all/any
kernels:

* non-contiguous layouts (transpose / permute / slice / expand) for the
  whole-tensor, single-dim and multi-dim reductions, including
  deterministic adversarial guard-buffer constructions in which a wrong
  read set necessarily flips the result;
* empty tensors ((0,), (0, 5), (5, 0)) x dim=None/0/1 (the old kernels
  crashed with ZeroDivisionError / math domain error);
* 0-dim tensors with dim=0/-1 (the old kernels rejected them);
* dim validation: illegal dims raise IndexError, duplicate dims raise
  RuntimeError, negative dims match native;
* keepdim=True/False combinations;
* dtype matrix including the uint8 output-dtype contract (ATen keeps
  uint8 for all/any);
* NaN / Inf / +-0 / smallest-subnormal handling;
* all-zero / single-nonzero / single-zero / sparse distributions.

References are PyTorch-native results computed on the CPU (importing
flag_gems does not patch torch.all/torch.any at the Python level). Where
the native MUSA device kernels deviate from upstream CPU semantics (int16
reduce, dim=[], dim=None + keepdim=True), the gems results are pinned
against the CPU reference instead.

Run with (torch_musa must be imported before flag_gems, which the shared
tests/conftest.py does not do):

    MUSA_VISIBLE_DEVICES=5 PYTHONPATH=src python -m pytest \
        tests/test_mthreads_boolean_reduction.py -v -p torch_musa
"""

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MTHREADS-specific boolean-reduction regression tests",
)

DEV = flag_gems.device

# Per-op marks so the usual `-m all` / `-m any` selection keeps working.
OPS = [
    pytest.param("all", marks=pytest.mark.all),
    pytest.param("any", marks=pytest.mark.any),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ref_op(op):
    return torch.all if op == "all" else torch.any


def _gems_op(op, entry):
    return getattr(flag_gems, op if entry == "whole" else f"{op}_{entry}")


def _assert_matches_reference(res, ref):
    assert isinstance(res, torch.Tensor)
    assert res.dtype == ref.dtype, f"dtype {res.dtype} != native {ref.dtype}"
    assert tuple(res.shape) == tuple(
        ref.shape
    ), f"shape {tuple(res.shape)} != native {tuple(ref.shape)}"
    assert torch.equal(
        res.cpu(), ref.cpu()
    ), f"value mismatch: gems={res.cpu().tolist()} native={ref.cpu().tolist()}"


def _check(op, entry, inp, **kwargs):
    """Run the gems entry point on the device input and compare value,
    dtype and shape with the CPU-native reference."""
    ref = _ref_op(op)(inp.cpu(), **kwargs)
    res = _gems_op(op, entry)(inp, **kwargs)
    _assert_matches_reference(res, ref)
    return res


def _mixed_data(shape, seed, zero_frac=0.35):
    """Deterministic float data with ~zero_frac zeros (mixed all/any lines)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(*shape, generator=g)
    base[torch.rand(*shape, generator=g) < zero_frac] = 0.0
    return base


# ---------------------------------------------------------------------------
# non-contiguous layouts (Round-1 risk 1 / B8)
# ---------------------------------------------------------------------------
# name -> (base shape, view builder applied on the DEVICE tensor, result ndim,
#          whether the built view is expected to be contiguous)
LAYOUTS = [
    ("contig2d", (4, 8), lambda t: t, 2, True),
    ("transpose2d", (4, 8), lambda t: t.t(), 2, False),  # (8,4) s(1,8)
    ("rowslice2d", (4, 10), lambda t: t[:, :8], 2, False),  # (4,8) s(10,1)
    ("stridedslice2d", (6, 10), lambda t: t[::2, ::3], 2, False),  # (3,4) s(20,3)
    ("expand2d", (1, 5), lambda t: t.expand(4, 5), 2, False),  # (4,5) s(0,1)
    ("strided1d", (16,), lambda t: t[::3], 1, False),  # (6,) s(3,)
    ("contig3d", (3, 4, 5), lambda t: t, 3, True),
    ("permute3d", (3, 4, 5), lambda t: t.permute(2, 0, 1), 3, False),  # (5,3,4)
    ("transpose023d", (2, 3, 4), lambda t: t.transpose(0, 2), 3, False),  # (4,3,2)
]
LAYOUT_NAMES = [name for name, _, _, _, _ in LAYOUTS]


def _layout_tensor(name, seed=42):
    for lname, shape, build, _, is_contig in LAYOUTS:
        if lname == name:
            inp = build(_mixed_data(shape, seed).to(DEV))
            assert inp.is_contiguous() == is_contig, (
                f"layout {name}: contiguity changed, test no longer covers "
                "the intended strided input"
            )
            return inp
    raise KeyError(name)


LAYOUT_DIM_CASES = [
    (name, dim) for name, _, _, ndim, _ in LAYOUTS for dim in range(ndim)
]

LAYOUT_DIMS_CASES = [
    (name, dims)
    for name, _, _, ndim, _ in LAYOUTS
    for dims in (
        [[0, 1], [1, 0], [0]]
        if ndim == 2
        else [[0]] if ndim == 1 else [[0, 1, 2], [2, 0], [0, 2], [-1, -2]]
    )
]


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("layout, dim", LAYOUT_DIM_CASES)
@pytest.mark.parametrize("op", OPS)
def test_noncontiguous_single_dim(op, layout, dim, keepdim):
    inp = _layout_tensor(layout)
    _check(op, "dim", inp, dim=dim, keepdim=keepdim)


@pytest.mark.parametrize("layout", LAYOUT_NAMES)
@pytest.mark.parametrize("op", OPS)
def test_noncontiguous_whole(op, layout):
    inp = _layout_tensor(layout)
    _check(op, "whole", inp)


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("layout, dims", LAYOUT_DIMS_CASES)
@pytest.mark.parametrize("op", OPS)
def test_noncontiguous_dims(op, layout, dims, keepdim):
    inp = _layout_tensor(layout)
    _check(op, "dims", inp, dim=dims, keepdim=keepdim)


# ---------------------------------------------------------------------------
# deterministic adversarial constructions (wrong read set => flipped result)
# ---------------------------------------------------------------------------
def _adv_transpose_all():
    # Native reference for this construction is [F, T, T, T].
    b = torch.ones(4, 8)
    b[0, 1] = 0.0
    return "all", "dim", {"dim": 0, "keepdim": False}, b.to(DEV).t()


def _adv_transpose_any():
    b = torch.zeros(4, 8)
    b[0, 1] = 7.0
    return "any", "dim", {"dim": 0, "keepdim": False}, b.to(DEV).t()


def _adv_expand_dim_all():
    # section 1.4 R1-e: rows 1-3 read out of the 5-element view
    buf = torch.full((40,), 7.0)
    buf[2] = 0.0
    x = buf.to(DEV)[:5].view(1, 5).expand(4, 5)
    return "all", "dim", {"dim": 1, "keepdim": False}, x


def _adv_expand_dim_any():
    buf = torch.zeros((40,))
    buf[2] = 7.0
    x = buf.to(DEV)[:5].view(1, 5).expand(4, 5)
    return "any", "dim", {"dim": 1, "keepdim": True}, x


def _adv_expand_whole_all():
    # marker sits outside the true element set (offsets 0..4) but inside the
    # 20-element linear span a buggy whole-reduction would read
    buf = torch.full((40,), 7.0)
    buf[10] = 0.0
    x = buf.to(DEV)[:5].view(1, 5).expand(4, 5)
    return "all", "whole", {}, x


def _adv_expand_whole_any():
    buf = torch.zeros((40,))
    buf[10] = 7.0
    x = buf.to(DEV)[:5].view(1, 5).expand(4, 5)
    return "any", "whole", {}, x


def _adv_slice_whole_all():
    # section 1.4 R1-b0: offset 1 is in a slice gap
    s = torch.ones(60)
    s[1] = 0.0
    return "all", "whole", {}, s.to(DEV).view(6, 10)[::2, ::3]


def _adv_slice_whole_any():
    s = torch.zeros(60)
    s[1] = 7.0
    return "any", "whole", {}, s.to(DEV).view(6, 10)[::2, ::3]


def _adv_slice_dims_any():
    s = torch.zeros(60)
    s[1] = 7.0
    return "any", "dims", {"dim": [0, 1]}, s.to(DEV).view(6, 10)[::2, ::3]


def _adv_anydims_transpose():
    # section 4.1 B8: single nonzero at offset 11 of the transposed view
    buf = torch.zeros(512)
    buf[11] = 7.0
    x = buf.to(DEV)[:32].view(4, 8).t()
    return "any", "dims", {"dim": [0, 1]}, x


def _adv_alldims_permute():
    buf = torch.ones(512)
    buf[7] = 0.0  # offset 7 is a true element of the permuted view
    x = buf.to(DEV)[:60].view(3, 4, 5).permute(2, 0, 1)
    return "all", "dims", {"dim": [0, 1, 2]}, x


ADV_CASES = [
    _adv_transpose_all,
    _adv_transpose_any,
    _adv_expand_dim_all,
    _adv_expand_dim_any,
    _adv_expand_whole_all,
    _adv_expand_whole_any,
    _adv_slice_whole_all,
    _adv_slice_whole_any,
    _adv_slice_dims_any,
    _adv_anydims_transpose,
    _adv_alldims_permute,
]


@pytest.mark.parametrize("case", ADV_CASES, ids=lambda fn: fn.__name__[4:])
def test_adversarial_guard_buffers(case):
    op, entry, kwargs, inp = case()
    assert not inp.is_contiguous()
    _check(op, entry, inp, **kwargs)


# ---------------------------------------------------------------------------
# empty tensors (Round-1 B1 / B2: the old kernels crashed on these)
# ---------------------------------------------------------------------------
EMPTY_SHAPES = [(0,), (0, 5), (5, 0)]


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("op", OPS)
def test_empty_whole(op, shape):
    inp = torch.empty(shape, device=DEV)
    res = _check(op, "whole", inp)
    # semantic ground truth: all() over an empty domain is vacuously True,
    # any() is False
    assert res.item() == (op == "all")


EMPTY_DIM_CASES = [
    ((0,), 0),
    ((0, 5), 0),
    ((0, 5), 1),
    ((5, 0), 0),
    ((5, 0), 1),
]


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("shape, dim", EMPTY_DIM_CASES)
@pytest.mark.parametrize("op", OPS)
def test_empty_dim(op, shape, dim, keepdim):
    inp = torch.empty(shape, device=DEV)
    _check(op, "dim", inp, dim=dim, keepdim=keepdim)


# ---------------------------------------------------------------------------
# 0-dim tensors (Round-1 B3: the old kernels rejected legal dim=0/-1)
# ---------------------------------------------------------------------------
ZERO_DIM_COMBOS = [
    pytest.param(None, False, id="dimNone-keepdimFalse"),
    pytest.param(0, False, id="dim0-keepdimFalse"),
    pytest.param(-1, False, id="dim-1-keepdimFalse"),
    pytest.param(0, True, id="dim0-keepdimTrue"),
    pytest.param(-1, True, id="dim-1-keepdimTrue"),
]


@pytest.mark.parametrize("value", [1.0, 0.0])
@pytest.mark.parametrize("dim, keepdim", ZERO_DIM_COMBOS)
@pytest.mark.parametrize("op", OPS)
def test_zero_dim_tensor(op, value, dim, keepdim):
    inp = torch.tensor(value, device=DEV)
    assert inp.ndim == 0
    _check(op, "dim", inp, dim=dim, keepdim=keepdim)


# ---------------------------------------------------------------------------
# dim validation (Round-1 B4 / B5: silent acceptance of illegal / duplicate dims)
# ---------------------------------------------------------------------------
INVALID_DIM_CASES = [
    ((4, 8), "dim", 5),
    ((4, 8), "dim", -3),
    ((4, 8), "dims", [0, 5]),
    ((4, 8), "dims", [0, -3]),
    ((6,), "dim", 1),
    ((), "dim", 1),
]


@pytest.mark.parametrize("shape, entry, dim", INVALID_DIM_CASES)
@pytest.mark.parametrize("op", OPS)
def test_invalid_dim_raises_index_error(op, shape, entry, dim):
    inp = _mixed_data(shape, 99).to(DEV) if shape else torch.tensor(1.0, device=DEV)
    with pytest.raises(IndexError):
        _ref_op(op)(inp.cpu(), dim=dim)
    with pytest.raises(IndexError):
        _gems_op(op, entry)(inp, dim=dim)


DUPLICATE_DIM_CASES = [[0, 0], [1, -1], [0, -2]]


@pytest.mark.parametrize("dims", DUPLICATE_DIM_CASES)
@pytest.mark.parametrize("op", OPS)
def test_duplicate_dims_raise_runtime_error(op, dims):
    inp = _mixed_data((4, 8), 98).to(DEV)
    with pytest.raises(RuntimeError):
        _ref_op(op)(inp.cpu(), dim=dims)
    with pytest.raises(RuntimeError):
        _gems_op(op, "dims")(inp, dim=dims)


# ---------------------------------------------------------------------------
# negative dims / dim-list semantics / keepdim
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dim", [-1, -2, -3])
@pytest.mark.parametrize("op", OPS)
def test_negative_dim(op, dim, keepdim):
    inp = _mixed_data((3, 4, 5), 77).to(DEV)
    _check(op, "dim", inp, dim=dim, keepdim=keepdim)


DIM_LISTS = [[0, 2], [2, 0], [1, 2], [0, 1, 2], [2, 1, 0], [-1, -2], (0, 2)]


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dims", DIM_LISTS)
@pytest.mark.parametrize("op", OPS)
def test_dim_lists(op, dims, keepdim):
    inp = _mixed_data((3, 4, 5), 88).to(DEV)
    _check(op, "dims", inp, dim=dims, keepdim=keepdim)


# ---------------------------------------------------------------------------
# dtype matrix (Round-1 B7: uint8 must keep uint8) and special values
# ---------------------------------------------------------------------------
DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    # int16 is exercised against the CPU reference: the native MUSA device
    # kernel crashes on int16 all/any reduce, but the gems result matches
    # upstream CPU semantics.
    torch.int16,
]

DTYPE_ENTRIES = [
    ("whole", {}),
    ("dim", {"dim": 0, "keepdim": False}),
    ("dims", {"dim": [0, 1], "keepdim": False}),
]


@pytest.mark.parametrize(
    "entry, kwargs", DTYPE_ENTRIES, ids=["whole", "dim0", "dims01"]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", OPS)
def test_dtype_matrix(op, dtype, entry, kwargs):
    base = _mixed_data((4, 8), 210, zero_frac=0.3)
    inp = (base != 0).to(DEV) if dtype is torch.bool else base.to(dtype).to(DEV)
    _check(op, entry, inp, **kwargs)


@pytest.mark.parametrize(
    "entry, kwargs", DTYPE_ENTRIES, ids=["whole", "dim0", "dims01"]
)
@pytest.mark.parametrize("op", OPS)
def test_uint8_output_dtype_contract(op, entry, kwargs):
    # ATen legacy: all/any keep uint8 for uint8 inputs, bool otherwise
    g = torch.Generator().manual_seed(211)
    inp = (torch.rand(4, 8, generator=g) > 0.3).to(torch.uint8).to(DEV)
    res = _check(op, entry, inp, **kwargs)
    assert res.dtype == torch.uint8
    assert set(res.cpu().unique().tolist()) <= {0, 1}


def _special_matrix(dtype):
    finfo = torch.finfo(dtype)
    subnormal = finfo.tiny * finfo.eps  # smallest subnormal (2^emin * 2^-mant)
    rows = [
        [float("nan"), 0.0],
        [float("inf"), float("-inf")],
        [float("nan"), 1.0],
        [0.0, -0.0],
        [subnormal, 0.0],
        [-subnormal, 1.0],
        [finfo.tiny, -0.0],
        [1.0, subnormal],
    ]
    return torch.tensor(rows, dtype=dtype).to(DEV)


SPECIAL_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SPECIAL_ENTRIES = [
    ("whole", {}),
    ("dim", {"dim": 0, "keepdim": False}),
    ("dim", {"dim": 1, "keepdim": True}),
    ("dims", {"dim": [0, 1], "keepdim": False}),
]


@pytest.mark.parametrize(
    "entry, kwargs", SPECIAL_ENTRIES, ids=["whole", "dim0", "dim1kd", "dims01"]
)
@pytest.mark.parametrize("dtype", SPECIAL_DTYPES)
@pytest.mark.parametrize("op", OPS)
def test_special_values(op, dtype, entry, kwargs):
    # NaN / +-Inf are nonzero; +-0 are zero; smallest subnormals are nonzero
    inp = _special_matrix(dtype)
    _check(op, entry, inp, **kwargs)


# ---------------------------------------------------------------------------
# value distributions
# ---------------------------------------------------------------------------
def _pattern_tensor(pattern):
    if pattern == "all_zero":
        return torch.zeros(4, 8)
    if pattern == "all_nonzero":
        return torch.ones(4, 8)
    if pattern == "single_zero":
        t = torch.ones(4, 8)
        t[1, 2] = 0.0
        return t
    if pattern == "single_nonzero":
        t = torch.zeros(4, 8)
        t[3, 5] = 1.0
        return t
    if pattern == "sparse_35pct":
        return _mixed_data((4, 8), 55)
    raise KeyError(pattern)


PATTERNS = ["all_zero", "all_nonzero", "single_zero", "single_nonzero", "sparse_35pct"]
PATTERN_ENTRIES = [
    ("whole", {}),
    ("dim", {"dim": 0, "keepdim": False}),
    ("dim", {"dim": 1, "keepdim": True}),
    ("dims", {"dim": [0, 1], "keepdim": False}),
]


@pytest.mark.parametrize(
    "entry, kwargs", PATTERN_ENTRIES, ids=["whole", "dim0", "dim1kd", "dims01"]
)
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("op", OPS)
def test_value_distributions(op, pattern, entry, kwargs):
    inp = _pattern_tensor(pattern).to(DEV)
    _check(op, entry, inp, **kwargs)


# ---------------------------------------------------------------------------
# cases where the native MUSA device kernels deviate from upstream CPU
# semantics (dim=[] becomes a full reduction; dim=None + keepdim=True raises
# "bad optional access"). The gems results follow upstream CPU semantics and
# are pinned here against the CPU reference.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("op", OPS)
def test_empty_dim_list_semantics(op):
    inp = _mixed_data((3, 4, 5), 66).to(DEV)
    _check(op, "dims", inp, dim=[], keepdim=False)


@pytest.mark.parametrize("op", OPS)
def test_whole_keepdim_true(op):
    inp = _mixed_data((3, 4, 5), 67).to(DEV)
    _check(op, "dim", inp, dim=None, keepdim=True)


# ---------------------------------------------------------------------------
# segcol route: segment-boundary isolation (regression for the cross-
# segment overshoot read). The segcol direct kernel streams each segment's
# rows in INNER-wide bands; when rows % INNER != 0 the last band must not
# read into the next segment (the liveness mask is bounded by the CTA's
# own row range). Both polarities are pinned; the input buffer is reused
# for the second polarity to halve peak memory.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [9, 16, 17])
@pytest.mark.parametrize("op", OPS)
def test_segcol_direct_segment_isolation(op, rows):
    # Imported lazily: the backend ops package requires torch_musa, which is
    # absent on non-MUSA runners where these tests are skipped at collection.
    from flag_gems.runtime.backend._mthreads.ops import _boolean_reduction as _br

    # (seg, rows, M) = (64, rows, 2^19): seg >= 64, M >= 512 and the
    # footprint stays >= 512 MiB for every rows value below, so the segcol
    # direct plan is guaranteed (smaller rows would fall below the
    # footprint gate and route to the fused kernel instead).
    shape = (64, rows, 1024, 512)
    probe = torch.empty(shape, dtype=torch.float16, device=DEV)
    plan = _br._plan_segcol(probe, [1])
    del probe
    assert plan is not None and plan[3]["direct"], "route changed"

    x = torch.ones(shape, dtype=torch.float16, device=DEV)
    x[1] = 0.0
    x[63] = 0.0
    _check(op, "dim", x, dim=1, keepdim=False)

    x.fill_(0.0)
    x[1] = 1.0
    x[63] = 1.0
    _check(op, "dim", x, dim=1, keepdim=True)
    del x
    torch.musa.empty_cache()


@pytest.mark.parametrize("op", OPS)
def test_segcol_split_odd_rows(op):
    from flag_gems.runtime.backend._mthreads.ops import _boolean_reduction as _br

    # Split mode (single segment): rows not a multiple of INNER. CTA row
    # bands are rounded up to a multiple of INNER, so the overshoot lanes
    # stay inside the padded band; pinned here against the CPU reference.
    shape = (524291, 512)
    probe = torch.empty(shape, dtype=torch.float16, device=DEV)
    plan = _br._plan_segcol(probe, [0])
    del probe
    assert plan is not None and not plan[3]["direct"], "route changed"

    x = torch.ones(shape, dtype=torch.float16, device=DEV)
    x[0] = 0.0
    x[262145] = 0.0
    x[-1] = 0.0
    _check(op, "dim", x, dim=0, keepdim=False)


# ---------------------------------------------------------------------------
# dispatch-boundary regressions added with the narrow-tile shallow-K any
# route, the deep-scan split floor and the M-dependent merge tiling:
# pin the boundary values of each gate (K around 8/256, tiles around 16,
# M around 16 for the merge tiling).
# ---------------------------------------------------------------------------
_SHALLOW_K = [7, 8, 9, 255, 256, 257]


@pytest.mark.parametrize("k", _SHALLOW_K)
@pytest.mark.parametrize("op", OPS)
def test_shallow_k_any_boundary(op, k):
    # any m-contiguous scans with 8 <= K <= 256 and M >= 64 take the
    # one-CTA-per-two-outputs route; both sides of each boundary must stay
    # correct.
    inp = _mixed_data((100, k, 64), 71).to(DEV)
    _check(op, "dims", inp, dim=[0, 1], keepdim=False)


@pytest.mark.parametrize("op", OPS)
def test_deep_scan_split_floor(op):
    # m-contiguous deep scans (K >= 65536) with few output tiles (<= 16)
    # raise the split count floor; [64, 1024, 8] fp32 sits exactly on the
    # K boundary with a single output tile.
    inp = _mixed_data((64, 1024, 8), 72).to(torch.float32).to(DEV)
    _check(op, "dims", inp, dim=[0, 1], keepdim=False)
    _check(op, "dims", inp, dim=[0, 1], keepdim=True)


@pytest.mark.parametrize("rows", [15, 16, 17])
@pytest.mark.parametrize("op", OPS)
def test_merge_tiling_boundary(op, rows):
    # The partials merge switches tile shape at M == 16; exercise the
    # boundary with a split-heavy scan (deep K, few outputs).
    inp = _mixed_data((rows, 4096, 8), 73).to(DEV)
    _check(op, "dims", inp, dim=[0, 1], keepdim=False)
