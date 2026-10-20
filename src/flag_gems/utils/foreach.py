"""Shared multi-tensor (foreach) execution utilities.

This module is the execution layer that PyTorch's ``aten::_foreach_*`` operators
need.  A ``Tensor[]`` is validated, grouped by ``(device, dtype)``, its pointers
and element counts are packed into a single device-side metadata buffer, and one
Triton kernel launch per group processes *all* tensors of that group.  The
per-element math is supplied by the caller as a ``triton.jit`` function passed
through ``tl.constexpr``, so a single executor kernel serves many operators
without an opcode switch.

Design notes
------------
* Launch count scales with the number of ``(device, dtype)`` groups, not with
  ``len(tensors)``.  ``tl.pointer_type`` needs a compile-time dtype, hence the
  grouping.
* Heterogeneous ``numel`` is handled by a *chunk table*: a flat array of
  ``(tensor index, element offset)`` pairs.  ``grid = number of chunks``, so
  every program does a bounded amount of work regardless of tensor sizes.
* Only non-overlapping-and-dense tensors can be addressed with flat indices.
  Gappy or overlapping views (``x[:, ::2]``, ``expand``) are staged through a
  contiguous copy, which costs extra launches; that cost is reported by
  :func:`launch_stats` rather than hidden.
"""

import array
import logging
import math
import numbers
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver as _triton_driver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# dtype plumbing
# ---------------------------------------------------------------------------

_TORCH_TO_TL = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.uint8: tl.uint8,
    torch.bool: tl.int1,
}

# A complex tensor is addressed as interleaved pairs of real components, so the
# kernel only ever needs the dtype of one component.
_COMPLEX_TO_REAL = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


def tl_dtype(dtype: torch.dtype):
    if dtype not in _TORCH_TO_TL:
        raise NotImplementedError(f"foreach executor does not support {dtype}")
    return _TORCH_TO_TL[dtype]


def real_dtype_of(dtype: torch.dtype) -> torch.dtype:
    """Result dtype of a complex-to-real elementwise op such as ``abs``."""
    return _COMPLEX_TO_REAL[dtype] if dtype.is_complex else dtype


def same_dtype(dtype: torch.dtype) -> torch.dtype:
    """Result dtype of a dtype-preserving elementwise op such as ``neg``."""
    return dtype


def int_to_float(dtype: torch.dtype) -> torch.dtype:
    """Result dtype of a transcendental op such as ``sin``.

    Measured against this PyTorch build: every integral and bool input is
    promoted to ``float32`` while floating and complex dtypes pass through.
    ``float16`` stays ``float16`` -- promotion is to the *default* float type
    only for non-float inputs, not to the widest one.
    """
    if dtype.is_floating_point or dtype.is_complex:
        return dtype
    return torch.float32


# ---------------------------------------------------------------------------
# validation, layout classification, grouping
# ---------------------------------------------------------------------------


def check_tensor_list(tensors: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Reproduce ATen's ``Tensor[]`` preconditions, including messages."""
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("argument 'self' must be tuple of Tensors")
    if len(tensors) == 0:
        raise RuntimeError("Tensor list must have at least one tensor.")
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("argument 'self' must be tuple of Tensors")
    return list(tensors)


def is_flat_addressable(t: torch.Tensor) -> bool:
    """Whether ``[data_ptr, data_ptr + numel)`` covers exactly the tensor.

    This is ``is_non_overlapping_and_dense``: the storage span equals ``numel``,
    which is precisely what flat indexing assumes.  A transposed but dense
    tensor qualifies -- its elements are visited in memory order, and the
    output allocated by ``empty_like`` carries the same stride, so the flat
    mapping stays element-wise correct.

    Computed here instead of calling ``torch.ops.aten`` to avoid a dispatcher
    round trip on the hot path; verified to agree with the ATen predicate on
    contiguous / transpose / slice / offset / permute / expand / gappy /
    empty / 0-dim / broadcast-squeezed cases.

    ``is_contiguous`` is checked first because it is a C-level call that answers
    the overwhelmingly common case; the Python stride walk below only runs for
    genuine views.  This ordering matters: the walk showed up as the single
    largest CPU cost when profiling a 128-tensor list.
    """
    if t.is_contiguous():
        return True
    if t.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted(
        (s, sz) for s, sz in zip(t.stride(), t.shape) if sz != 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def has_internal_overlap(t: torch.Tensor) -> bool:
    """True when several logical elements provably alias one memory location.

    ``_debug_has_internal_overlap`` is tri-state: ``0`` = no overlap, ``1`` =
    overlap, ``2`` = too hard to tell.  Only ``1`` must reject an in-place
    write; PyTorch itself accepts gappy-but-disjoint views such as
    ``x[:, ::2]`` (which report ``2``) and rejects only ``expand``-style
    aliasing.
    """
    return torch._debug_has_internal_overlap(t) == 1


def group_by_device_dtype(
    tensors: Sequence[torch.Tensor],
) -> Dict[Tuple[Any, torch.dtype], List[int]]:
    """Group tensor *indices* by ``(device, dtype)``, preserving input order."""
    groups: Dict[Tuple[Any, torch.dtype], List[int]] = {}
    for i, t in enumerate(tensors):
        groups.setdefault((t.device, t.dtype), []).append(i)
    return groups


def _pick_block(numels: Sequence[int], max_programs: int = 1 << 18) -> int:
    """Block size balancing per-program work against the 2D grid size.

    The grid is ``(len(numels), cdiv(max numel, block))``, so tensors smaller
    than the largest one contribute programs that exit on the mask.  Those cost
    almost nothing individually, but a very skewed list still needs a larger
    block to keep the total program count in check.

    The search starts at 1024 rather than at the smallest workable value.  Each
    program pays three scalar ``int64`` metadata loads before it touches any
    data, so a small block amortises that fixed cost over too little work: on
    128 x 1M fp32 a sweep measured 339us at ``BLOCK=1024`` against 434us at 256,
    and the ordering held across every shape tried.  1024 elements is also one
    element per lane at ``num_warps=4``, which is why larger blocks stop helping.
    """
    nt = len(numels)
    max_numel = max(numels)
    for block in (1024, 2048, 4096, 8192, 16384):
        if nt * ((max_numel + block - 1) // block) <= max_programs:
            return block
    return 16384


# ---------------------------------------------------------------------------
# executor kernels
# ---------------------------------------------------------------------------


@triton.jit
def foreach_unary_kernel(
    meta_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    IN_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch, ``NT`` tensors.

    ``meta_ptr`` layout, all int64::

        [0,      NT)     input pointers
        [NT,   2*NT)     output pointers
        [2*NT, 3*NT)     element counts

    The grid is 2D, ``(NT, max chunks per tensor)``: axis 0 selects the tensor
    and axis 1 the chunk within it.  Programs past a tensor's own element count
    are fully masked out, which costs a load and an exit -- cheaper than
    materializing an explicit chunk table on the CPU, which measured as the
    single largest overhead in the end-to-end benchmark.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(IN_DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 2 * NT + t)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    x = tl.load(in_ptr + idx, mask=mask, other=0)
    # The value is promoted to the output dtype *before* the math runs, which is
    # what ``pointwise_dynamic`` does for its scalar functions.  Those functions
    # commonly end in ``.to(x.dtype)``, so handing them an unpromoted integer
    # would truncate the result: ``log1p`` on int64 measured 0.693 of error
    # until the promotion was moved ahead of the call.
    tl.store(out_ptr + idx, fn(x.to(OUT_DT)).to(OUT_DT), mask=mask)


@triton.jit
def foreach_unary_c2r_kernel(
    meta_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    R_DT: tl.constexpr,
    ACC_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Complex input, real output; ``fn(re, im)`` returns one real value.

    The complex input is read through a real-typed pointer where logical
    element ``i`` occupies slots ``2*i`` (real) and ``2*i + 1`` (imaginary).
    Triton's dtype tables have no complex entries, so this interleaved view is
    what makes complex support possible at all.  Metadata layout and 2D grid
    match :func:`foreach_unary_kernel`; counts are in *complex* elements.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(R_DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(R_DT))
    n_elements = tl.load(meta_ptr + 2 * NT + t)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    re = tl.load(in_ptr + 2 * idx, mask=mask, other=0).to(ACC_DT)
    im = tl.load(in_ptr + 2 * idx + 1, mask=mask, other=0).to(ACC_DT)
    tl.store(out_ptr + idx, fn(re, im).to(R_DT), mask=mask)


@triton.jit
def foreach_unary_c2c_kernel(
    meta_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    R_DT: tl.constexpr,
    ACC_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Complex input, complex output; ``fn(re, im)`` returns a ``(re, im)`` pair.

    Same interleaved addressing as :func:`foreach_unary_c2r_kernel`, but both
    components are written back.  Complex support has to live here rather than
    be borrowed from the single-tensor operators: those raise
    ``KeyError: 'complex64'`` from Triton's dtype table, yet PyTorch's foreach
    variants of ``exp``/``log``/``sin``/... do accept complex, so parity
    requires a dedicated path.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(R_DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(R_DT))
    n_elements = tl.load(meta_ptr + 2 * NT + t)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    re = tl.load(in_ptr + 2 * idx, mask=mask, other=0).to(ACC_DT)
    im = tl.load(in_ptr + 2 * idx + 1, mask=mask, other=0).to(ACC_DT)
    out_re, out_im = fn(re, im)
    tl.store(out_ptr + 2 * idx, out_re.to(R_DT), mask=mask)
    tl.store(out_ptr + 2 * idx + 1, out_im.to(R_DT), mask=mask)


@triton.jit
def foreach_binary_kernel(
    meta_ptr,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    B_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    ALPHA: tl.constexpr,
    B_SCALAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch, ``NT`` tensors, two tensor inputs per element.

    ``meta_ptr`` layout, all int64::

        [0,      NT)     first input pointers
        [NT,   2*NT)     second input pointers
        [2*NT, 3*NT)     output pointers
        [3*NT, 4*NT)     element counts

    The second operand is read with its *own* pointer rather than being folded
    into the first, which is what makes ``.List`` correct for tensors whose
    strides differ: the two lists are paired position by position, and each
    element count is that pair's own. A unary operator can get away with flat
    traversal of a non-contiguous tensor because the element *set* is unchanged,
    but pairing two differently-strided tensors by flat index would silently
    match up the wrong elements, so both sides are staged to dense form by the
    caller before they reach here.

    ``B_SCALAR`` is the ``.Tensor`` broadcast case: the second operand holds a
    single element, so it is loaded once per program and splatted by the
    element-wise function instead of being materialised into a full-size copy
    on the host.  Reading it here is a uniform device load; reading it back on
    the host with ``.item()`` would synchronise and drain every queued launch.

    ``ALPHA`` scales the second operand before the element-wise function runs;
    it is a ``constexpr`` so the multiply compiles away for the default value.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(A_DT))
    b_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(B_DT))
    out_ptr = tl.load(meta_ptr + 2 * NT + t).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 3 * NT + t)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    a = tl.load(a_ptr + idx, mask=mask, other=0)
    if B_SCALAR:
        b = tl.load(b_ptr)
    else:
        b = tl.load(b_ptr + idx, mask=mask, other=0)
    b = (b * ALPHA).to(OUT_DT) if ALPHA != 1 else b.to(OUT_DT)
    tl.store(out_ptr + idx, fn(a.to(OUT_DT), b).to(OUT_DT), mask=mask)


@triton.jit
def foreach_binary_scalar_kernel(
    meta_ptr,
    scalar_ptr,
    scalar_value: tl.float64,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    S_DT: tl.constexpr,
    ALPHA: tl.constexpr,
    SCALAR_BY_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``NT`` tensors against one scalar *each*.

    ``meta_ptr`` layout matches the unary kernel (in / out / counts); the
    per-tensor scalars live in their own ``scalar_ptr`` array indexed by tensor.

    ``SCALAR_BY_VALUE`` selects the ``.Scalar`` path: when every tensor shares
    the same value there is nothing to index, so it travels as an ordinary
    kernel argument and no device buffer is built.  ``scalar_value`` is declared
    ``float64`` so the ``.Scalar`` path keeps the exact value the device buffer
    would have held; the cast to ``S_DT`` below reproduces the host-side
    ``torch.tensor(..., dtype=S_DT)`` rounding bit for bit.  ``.ScalarList``
    still needs the per-tensor array.

    ``ALPHA`` scales the scalar before the element-wise function runs; it is a
    ``constexpr`` so the multiply compiles away for the default value.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(A_DT))
    out_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 2 * NT + t)
    if SCALAR_BY_VALUE:
        s = scalar_value.to(S_DT)
    else:
        s = tl.load(scalar_ptr + t)
    s = (s * ALPHA).to(S_DT) if ALPHA != 1 else s
    s = s.to(OUT_DT)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    a = tl.load(a_ptr + idx, mask=mask, other=0)
    tl.store(out_ptr + idx, fn(a.to(OUT_DT), s.to(OUT_DT)).to(OUT_DT), mask=mask)


@triton.jit
def foreach_ternary_kernel(
    meta_ptr,
    scalar_ptr,
    scalar_value: tl.float64,
    fn: tl.constexpr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    S_DT: tl.constexpr,
    SCALAR_BY_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``NT`` tensors, three tensor inputs and one per-tensor scalar.

    ``meta_ptr`` layout, all int64::

        [0,      NT)     self pointers
        [NT,   2*NT)     tensor1 pointers
        [2*NT, 3*NT)     tensor2 pointers
        [3*NT, 4*NT)     output pointers
        [4*NT, 5*NT)     element counts

    This serves ``addcmul``/``addcdiv`` (``self + value * t1 <op> t2``) and the
    ``lerp`` weight forms; the scalar array is the ``value``/``weight``
    argument, broadcast on the host when the overload supplies a single value.

    ``SCALAR_BY_VALUE`` selects the ``.Scalar`` form: a single value that every
    tensor shares travels as an ordinary kernel argument instead of a device
    buffer.  ``scalar_value`` is declared ``float64`` so nothing is lost on the
    way in; the cast to ``S_DT`` reproduces the buffer's rounding exactly.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(A_DT))
    b_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(A_DT))
    c_ptr = tl.load(meta_ptr + 2 * NT + t).to(tl.pointer_type(A_DT))
    out_ptr = tl.load(meta_ptr + 3 * NT + t).to(tl.pointer_type(OUT_DT))
    n_elements = tl.load(meta_ptr + 4 * NT + t)
    if SCALAR_BY_VALUE:
        s = scalar_value.to(S_DT)
    else:
        s = tl.load(scalar_ptr + t)

    idx = offset + tl.arange(0, BLOCK)
    mask = idx < n_elements
    a = tl.load(a_ptr + idx, mask=mask, other=0).to(OUT_DT)
    b = tl.load(b_ptr + idx, mask=mask, other=0).to(OUT_DT)
    c = tl.load(c_ptr + idx, mask=mask, other=0).to(OUT_DT)
    tl.store(out_ptr + idx, fn(a, b, c, s.to(OUT_DT)).to(OUT_DT), mask=mask)


# ---------------------------------------------------------------------------
# launch accounting (used by the launch-count regression test)
# ---------------------------------------------------------------------------


class _LaunchStats:
    """Counters describing how the last call was decomposed.

    Kept as plain integers so a test can assert that executor launches track
    the number of ``(device, dtype)`` groups instead of ``len(tensors)``.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.executor_launches = 0
        self.groups = 0
        self.staged_inputs = 0
        self.writebacks = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "executor_launches": self.executor_launches,
            "groups": self.groups,
            "staged_inputs": self.staged_inputs,
            "writebacks": self.writebacks,
        }


_STATS = _LaunchStats()


def launch_stats() -> Dict[str, int]:
    """Snapshot of the counters from the most recent foreach call."""
    return _STATS.as_dict()


# Every argument of the executor kernels other than the metadata pointer is a
# ``tl.constexpr``, so ``(kernel, NT, fn, dtype, dtype, BLOCK)`` identifies the
# compiled variant exactly and a replay only has to hand the same values to the
# kernel's launcher.  Triton's Python dispatch between that lookup and the
# actual launch -- binder, cache-key hash, launch-metadata and hook checks --
# measured ~4us per launch, about a tenth of the small-shape budget, and none of
# it can change the result.  The hit rate is high by construction: a benchmark
# loop launches the same variant thousands of times.
#
# Only the *launcher* is cached.  The metadata tensor is rebuilt and uploaded on
# every call, so no device-visible buffer is shared across calls; see
# ``_device_meta``.
#
# The only thing Triton specializes on here is the metadata pointer's 16-byte
# alignment, so that bit is part of the key and an unusually aligned allocation
# gets a variant of its own rather than a mismatch.  A miss, an installed launch
# hook (proton and other instrumentation install one), or a launcher that raises
# all take the normal path, which fills the cache.
_LAUNCH_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any, Any]] = {}

# Replay accounting.  ``replay`` counts calls served by the cached launcher,
# ``dispatch`` counts calls that went through Triton's dispatcher (first call
# per key, or a fallback).  Together they equal the number of group launches,
# which is what makes a "the cache is write-only" bug visible without a probe.
_LAUNCH_STATS: Dict[str, int] = {"replay": 0, "dispatch": 0}


def launch_cache_stats() -> Dict[str, int]:
    """Cumulative hits/misses of the executor launch cache (diagnostics only).

    Deliberately separate from :func:`launch_stats`, which is a per-call
    snapshot used by the tests.
    """
    return dict(_LAUNCH_STATS)


def _launch_executor_kernel(
    kernel: Callable,
    grid: Tuple[int, int],
    nt: int,
    meta: torch.Tensor,
    fn: Callable,
    dt_in: Any,
    dt_out: Any,
    block: int,
) -> None:
    # ``meta.device`` is part of the key: a compiled kernel belongs to one
    # device's module, so replaying device A's launcher for device B would use
    # the wrong context.
    key = (
        kernel,
        meta.device,
        nt,
        fn,
        dt_in,
        dt_out,
        block,
        not (meta.data_ptr() & 15),
    )
    entry = _LAUNCH_CACHE.get(key)
    runtime = knobs.runtime
    # ``launch_enter_hook`` is a ``HookChain`` in triton 3.6, never ``None``:
    # the chain exists whether or not anyone subscribed.  ``.calls`` is the
    # list of registered callbacks, so an empty list is the real "no hooks"
    # state.  Proton and other instrumentation add to it, and a replay would
    # then skip their notifications.
    if (
        entry is not None
        and not runtime.launch_enter_hook.calls
        and not runtime.launch_exit_hook.calls
    ):
        try:
            entry[0](
                grid[0],
                grid[1],
                1,
                _triton_driver.active.get_current_stream(meta.device.index),
                entry[1],
                entry[2],
                None,
                None,
                None,
                meta,
                fn,
                nt,
                dt_in,
                dt_out,
                block,
            )
            _LAUNCH_STATS["replay"] += 1
            return
        except Exception:
            # A launcher that cannot be replayed this way is not a correctness
            # problem: forget the entry and let Triton dispatch as usual.
            _LAUNCH_CACHE.pop(key, None)
    _LAUNCH_STATS["dispatch"] += 1
    compiled = kernel[grid](meta, fn, nt, dt_in, dt_out, block)
    if compiled is not None:
        _LAUNCH_CACHE[key] = (
            compiled.run,
            compiled.function,
            compiled.packed_metadata,
        )


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def _launch_group(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    fn: Callable,
    in_dtype: torch.dtype,
    device: Any,
    is_complex: bool,
) -> None:
    numels = [t.numel() for t in srcs]
    max_numel = max(numels)
    if max_numel == 0:
        return
    block = _pick_block(numels)

    nt = len(srcs)
    # One flat Python list -> one tensor -> one H2D transfer.  Building this as
    # three separate tensors and slicing them in measured noticeably slower,
    # because each slice assignment is its own copy.
    #
    # An in-place call passes the same list object as both sources and
    # destinations, so the pointer run is read once instead of twice.
    #
    # A pinned staging pool was tried here and rejected.  In isolation the pinned
    # copy is 1.7-3.8x faster than this pageable one, but reusing pinned buffers
    # without ordering the host refill against the outstanding device read is a
    # real data race (a probe that forced the window open corrupted 21 of 24
    # calls).  Making it safe needs a CUDA event per call, and allocating plus
    # recording that event costs more than the pageable copy saves: end to end it
    # measured 93us per call against 46us for this version.
    ptrs = [t.data_ptr() for t in srcs]
    if dsts is srcs:
        values = ptrs + ptrs + numels
    else:
        values = ptrs + [t.data_ptr() for t in dsts] + numels
    meta = _device_meta(values, device)

    grid = (nt, triton.cdiv(max_numel, block))
    if is_complex:
        real_dtype = _COMPLEX_TO_REAL[in_dtype]
        acc = tl.float32 if real_dtype is not torch.float64 else tl.float64
        kernel = (
            foreach_unary_c2r_kernel
            if dsts[0].dtype.is_floating_point
            else foreach_unary_c2c_kernel
        )
        _launch_executor_kernel(
            kernel, grid, nt, meta, fn, tl_dtype(real_dtype), acc, block
        )
    else:
        _launch_executor_kernel(
            foreach_unary_kernel,
            grid,
            nt,
            meta,
            fn,
            tl_dtype(in_dtype),
            tl_dtype(dsts[0].dtype),
            block,
        )
    _STATS.executor_launches += 1
    _STATS.groups += 1


def check_dtype_supported(
    dtype: torch.dtype,
    allowed: Optional[frozenset],
    message: Optional[str] = None,
    error_type: type = RuntimeError,
) -> None:
    """Reject dtypes the corresponding ATen operator also rejects.

    ``allowed=None`` means "no restriction".  Each operator's set was measured
    against this PyTorch build rather than assumed, because the exclusions are
    not guessable: ``_foreach_floor`` rejects ``bool`` but accepts every integer
    dtype, ``_foreach_frac`` rejects all integers, and ``_foreach_sign``
    accepts ``bool`` while rejecting complex.

    ``message`` overrides the generic text for operators whose ATen refusal is
    worded differently, such as ``sub`` on bool.  ``error_type`` matches the
    class ATen raises, which for ``pow`` on bool is ``NotImplementedError``.
    """
    if allowed is not None and dtype not in allowed:
        if message is not None:
            raise error_type(message)
        raise error_type(
            f'"foreach" not implemented for \'{str(dtype).replace("torch.", "")}\''
        )


# ``flag_gems`` overrides ``Tensor.copy_`` on the CUDA key inside
# ``use_gems()``, and ``_device_meta`` runs that copy once per group.  The gems
# wrapper validates its arguments and then re-dispatches this CPU->CUDA copy to
# the composite aten kernel anyway, so address that implementation directly.
# The handle is optional: a build without ``redispatch`` keeps the method.
try:
    _ATEN_COMPOSITE_KEYSET = torch._C.DispatchKeySet(
        torch._C.DispatchKey.CompositeExplicitAutograd
    )
except (AttributeError, RuntimeError):  # pragma: no cover - depends on torch build
    _ATEN_COMPOSITE_KEYSET = None

try:
    _ATEN_COPY_REDISPATCH = torch.ops.aten.copy_.default.redispatch
except (AttributeError, RuntimeError):  # pragma: no cover - depends on torch build
    _ATEN_COPY_REDISPATCH = None


def foreach_unary(
    tensors: Sequence[torch.Tensor],
    fn: Callable,
    *,
    complex_fn: Optional[Callable] = None,
    out_dtype_fn: Callable[[torch.dtype], torch.dtype] = same_dtype,
    inplace: bool = False,
    allowed_dtypes: Optional[frozenset] = None,
) -> List[torch.Tensor]:
    """Apply a unary elementwise ``triton.jit`` function to a whole TensorList.

    Args:
        tensors: the ``Tensor[]`` argument as received from ATen.
        fn: ``@triton.jit`` function of one value, used for real dtypes.
        complex_fn: ``@triton.jit`` function of ``(re, im)`` for complex
            inputs.  ``None`` means complex is unsupported by this operator.
        out_dtype_fn: maps input dtype to output dtype.
        inplace: write results back into ``tensors`` and return them.
        allowed_dtypes: dtypes the operator accepts; ``None`` means all.

    Returns:
        A list of result tensors, one per input, in input order.
    """
    tensors = check_tensor_list(tensors)
    _STATS.reset()

    # Fast path: one device, one dtype.  Bucketing exists for the mixed case,
    # and its dict/tuple bookkeeping is the largest per-tensor cost in this
    # walk; every unary operator here is normally called that way, so pay for
    # the general machinery only when the list actually mixes.
    first = tensors[0]
    device = first.device
    dtype = first.dtype
    if allowed_dtypes is not None and dtype not in allowed_dtypes:
        check_dtype_supported(dtype, allowed_dtypes)
    if dtype.is_complex and complex_fn is None:
        raise NotImplementedError(
            f"foreach executor has no complex kernel for {dtype}"
        )
    out_dtype = out_dtype_fn(dtype)
    if inplace and out_dtype != dtype:
        # Same refusal as ATen: an in-place op may not silently narrow its
        # promoted result back into the input's dtype.
        raise RuntimeError(
            f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
            " can't be cast to the desired output type "
            f"{str(dtype).replace('torch.', '').capitalize()}"
        )
    is_complex = dtype.is_complex
    kernel_fn = complex_fn if is_complex else fn
    results: List[torch.Tensor] = []
    srcs: List[torch.Tensor] = []
    dsts: List[torch.Tensor] = []
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for t in tensors:
        if t.dtype != dtype or t.device != device:
            # Mixed list: redo the call through the bucketing path.  The
            # counters are reset so staging done above is not counted twice.
            _STATS.reset()
            return _foreach_unary_grouped(
                tensors,
                fn,
                complex_fn=complex_fn,
                out_dtype_fn=out_dtype_fn,
                inplace=inplace,
                allowed_dtypes=allowed_dtypes,
            )
        flat = is_flat_addressable(t)
        if inplace:
            # A flat-addressable tensor is non-overlapping-and-dense by
            # definition, so the dispatcher-backed probe cannot change the
            # answer there; skipping it saves a dispatch per tensor.
            if not flat and has_internal_overlap(t):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            if flat:
                src = t
            else:
                # Gappy view: compute on a dense copy, then scatter back.
                src = t.contiguous()
                writebacks.append((t, src))
                _STATS.staged_inputs += 1
            results.append(t)
        else:
            src = t if flat else t.contiguous()
            if src is not t:
                _STATS.staged_inputs += 1
            # empty_like reproduces PyTorch's foreach output stride policy: it
            # keeps a dense tensor's stride (so transposed inputs give
            # transposed outputs) and normalizes everything else to contiguous.
            # One allocation per tensor, not one packed buffer plus unbind, is
            # also what keeps aliasing observable-equivalent to ATen: no two
            # outputs of a call share a storage (checked on two 4x4,
            # mixed-shape, transposed, zero-dim and one-element lists).
            dst = torch.empty_like(src, dtype=out_dtype)
            results.append(dst)
            dsts.append(dst)
        srcs.append(src)

    if inplace:
        # Same list object: ``_launch_group`` then knows the pointers occur
        # twice in a row and reads each ``data_ptr`` once.  Returning the
        # inputs themselves means the only aliasing is what the caller passed
        # in (a tensor repeated twice stays shared), as with ATen.
        dsts = srcs
    _launch_group(srcs, dsts, kernel_fn, dtype, device, is_complex)

    for dst_view, staged in writebacks:
        dst_view.copy_(staged)
        _STATS.writebacks += 1

    return results


def _foreach_unary_grouped(
    tensors: Sequence[torch.Tensor],
    fn: Callable,
    *,
    complex_fn: Optional[Callable] = None,
    out_dtype_fn: Callable[[torch.dtype], torch.dtype] = same_dtype,
    inplace: bool = False,
    allowed_dtypes: Optional[frozenset] = None,
) -> List[torch.Tensor]:
    """Bucketing path for a TensorList that mixes devices or dtypes.

    ``tensors`` is already validated and normalised to a list.
    """
    results: List[Optional[torch.Tensor]] = [None] * len(tensors)
    # Buckets keyed by (device, input dtype); values are aligned src/dst lists.
    buckets: Dict[Tuple[Any, torch.dtype], Tuple[List, List]] = {}
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for i, t in enumerate(tensors):
        dtype = t.dtype
        # The helper is only a message builder; calling it per tensor for the
        # accepted case measured 0.6us on a sixteen-tensor list.
        if allowed_dtypes is not None and dtype not in allowed_dtypes:
            check_dtype_supported(dtype, allowed_dtypes)
        if dtype.is_complex and complex_fn is None:
            raise NotImplementedError(
                f"foreach executor has no complex kernel for {dtype}"
            )
        out_dtype = out_dtype_fn(dtype)
        key = (t.device, dtype)
        flat = is_flat_addressable(t)

        if inplace:
            if out_dtype != dtype:
                # Same refusal as ATen: an in-place op may not silently narrow
                # its promoted result back into the input's dtype.
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(dtype).replace('torch.', '').capitalize()}"
                )
            # A flat-addressable tensor is non-overlapping-and-dense by
            # definition, so the dispatcher-backed probe cannot change the
            # answer there; skipping it saves a dispatch per tensor.
            if not flat and has_internal_overlap(t):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            if flat:
                src = dst = t
            else:
                # Gappy view: compute on a dense copy, then scatter back.
                src = dst = t.contiguous()
                writebacks.append((t, src))
                _STATS.staged_inputs += 1
            results[i] = t
        else:
            src = t if flat else t.contiguous()
            if src is not t:
                _STATS.staged_inputs += 1
            # empty_like reproduces PyTorch's foreach output stride policy: it
            # keeps a dense tensor's stride (so transposed inputs give
            # transposed outputs) and normalizes everything else to contiguous.
            dst = torch.empty_like(src, dtype=out_dtype)
            results[i] = dst

        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = ([], [])
        bucket[0].append(src)
        bucket[1].append(dst)

    for (device, dtype), (srcs, dsts) in buckets.items():
        kernel_fn = complex_fn if dtype.is_complex else fn
        _launch_group(srcs, dsts, kernel_fn, dtype, device, dtype.is_complex)

    for dst_view, staged in writebacks:
        dst_view.copy_(staged)
        _STATS.writebacks += 1

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# multi-input entry points
# ---------------------------------------------------------------------------


def _dense(t: torch.Tensor) -> torch.Tensor:
    """A flat-addressable view of ``t``, copying only when necessary.

    Unlike the unary path, a merely *dense* tensor is not enough for the paired
    kernels: two tensors that are each dense but carry different strides would
    be matched up element-by-element in the wrong order under flat indexing.
    ``contiguous()`` is therefore the condition here, not
    ``is_non_overlapping_and_dense``.
    """
    if t.is_contiguous():
        return t
    _STATS.staged_inputs += 1
    return t.contiguous()


def _check_same_length(a: Sequence[torch.Tensor], b: Sequence[Any], what: str) -> None:
    if len(a) != len(b):
        raise RuntimeError(
            f"Tensor list must have same number of elements as {what}, "
            f"got {len(a)} and {len(b)}"
        )


# Promotion is a pure function of the operand dtypes (plus whether an operand is
# zero-dimensional, which ATen treats as a weak scalar), so the answer is
# memoised: a 16-tensor list asks the same question 16 times.
_PROMOTION_CACHE: Dict[Tuple[Any, ...], torch.dtype] = {}


def _promotion_probe(t: torch.Tensor) -> torch.Tensor:
    """A meta tensor standing in for ``t`` in a ``result_type`` query."""
    if t.dim() == 0:
        return torch.empty((), dtype=t.dtype, device="meta")
    return torch.empty(0, dtype=t.dtype, device="meta")


def _promote_dtypes(
    a: torch.Tensor, b: torch.Tensor, c: Optional[torch.Tensor] = None
) -> torch.dtype:
    """Two- or three-way ``result_type`` fold, memoised on the operand shapes.

    ``result_type`` takes two operands, so a three-way promotion is a fold with
    an intermediate dtype.  The original spelled that intermediate out as a
    fresh ``meta`` tensor per tensor in the list, which measured 54us for a
    16-tensor ``addcmul`` -- more than the kernel it feeds.  Nothing about the
    answer depends on the values, so the dtype tuple keys a dict instead.
    """
    key = (
        a.dtype,
        a.dim() == 0,
        b.dtype,
        b.dim() == 0,
        c.dtype if c is not None else None,
        c is not None and c.dim() == 0,
    )
    out = _PROMOTION_CACHE.get(key)
    if out is None:
        out = torch.result_type(_promotion_probe(a), _promotion_probe(b))
        if c is not None:
            out = torch.result_type(
                torch.empty(0, dtype=out, device="meta"), _promotion_probe(c)
            )
        _PROMOTION_CACHE[key] = out
    return out


def _promote_with_scalar(dtype: torch.dtype, value: Any) -> torch.dtype:
    """``result_type(tensor, scalar)`` memoised on ``(dtype, type(value))``.

    A Python number is a weak scalar, so its *value* never changes the answer --
    only its type (an ``int`` keeps an integral tensor integral, a ``float``
    promotes it to the default float dtype).  A zero-dimensional tensor is also
    weak, so the probe only has to match that one property.
    """
    zero_dim = isinstance(value, torch.Tensor) and value.dim() == 0
    key = ("scalar", dtype, type(value), zero_dim)
    out = _PROMOTION_CACHE.get(key)
    if out is None:
        probe = torch.empty(0, dtype=dtype, device="meta")
        out = torch.result_type(probe, value)
        _PROMOTION_CACHE[key] = out
    return out


def _common_scalar(vals: Sequence[Any]) -> Optional[Any]:
    """The single value every slot shares, or ``None`` when they differ.

    ``==`` on its own would fold ``0.0`` and ``-0.0`` together; the sign bit
    matters because ``div``/``pow`` turn it into ``+inf`` versus ``-inf``.
    NaNs never compare equal, so they fall through to the per-tensor buffer.
    """
    # ``float`` is what the device-buffer path would have applied anyway, and
    # it keeps the by-value argument a plain number even if a caller handed in
    # zero-dimensional tensors.
    first = float(vals[0])
    if first == 0.0:
        sign = math.copysign(1.0, first)
        for v in vals:
            v = float(v)
            if v != first or math.copysign(1.0, v) != sign:
                return None
        return first
    # Outside the zeros the value itself carries the sign, so the explicit
    # sign-bit test is only needed for ±0.0.
    for v in vals:
        if float(v) != first:
            return None
    return first


def _device_meta(values: List[int], device: Any) -> torch.Tensor:
    """A fresh int64 device tensor holding ``values``.

    ``array.array`` + ``frombuffer`` skips the generic sequence-to-tensor path,
    and going through ``empty`` + ``copy_`` sidesteps the dispatcher's
    ``_to_copy``: inside ``use_gems()`` the plain ``torch.tensor(...).to(device)``
    form runs the gems implementation and measured 21.5us against 13.1us here.

    The copy addresses the composite aten implementation directly.  Inside
    ``use_gems()`` the ``Tensor.copy_`` method runs flag_gems' Python wrapper,
    which validates its arguments and then redispatches this CPU->CUDA copy to
    exactly that composite kernel anyway; the wrapper measured ~10us per call
    (P1's cProfile: ``copy_`` cumulative 13.5-14.9us, of which the transfer is
    small).  ``redispatch`` is only an optimisation, so a build without it falls
    back to the regular method.

    Both sides are fresh on every call.  The host buffer is pageable, so the
    runtime stages it before the copy call returns; nothing is reused, which is
    exactly what made the cached pinned staging pool unsafe.
    """
    host = torch.frombuffer(array.array("q", values), dtype=torch.int64)
    out = torch.empty(len(values), dtype=torch.int64, device=device)
    if _ATEN_COPY_REDISPATCH is None or _ATEN_COMPOSITE_KEYSET is None:
        out.copy_(host, non_blocking=True)
    else:
        # Skip the Python-level copy_ registration and go straight to the
        # CompositeExplicitAutograd implementation.
        _ATEN_COPY_REDISPATCH(_ATEN_COMPOSITE_KEYSET, out, host, True)
    return out


def _resolve_scalars(
    tensors: Sequence[torch.Tensor], scalars: Any
) -> Tuple[List[Any], Optional[float], Optional[type]]:
    """Normalise the ``.Scalar`` / ``.ScalarList`` / ``.Tensor`` forms into a list.

    The ``.Tensor`` overloads of ``addcmul``/``addcdiv`` take the values as a
    one-dimensional tensor holding one entry per tensor in the list (ATen
    requires it on the CPU), so it is unpacked here rather than being mistaken
    for a single value.

    Returns ``(values, common, vclass)``: one entry per tensor, plus the value
    every entry shares as a ``float`` -- or ``None`` when a scan is needed to
    find out, or when the value is not a real number and cannot be converted --
    plus the Python type of every entry when the broadcast form makes it one
    object repeated (``None`` when it has to be looked at per element).  A plain
    broadcast scalar is already its own common value, so the equality scan that
    ``_common_scalar`` performs can be skipped for it.

    ``float`` conversion is limited to ``numbers.Real`` so that an illegal
    complex scalar keeps the error the base implementation produced: the
    launcher's ``_common_scalar`` raises ``TypeError`` for it, and for an
    all-empty tensor list -- where the launch is skipped -- the call returns
    the promoted empty outputs instead of failing early here.
    """
    if isinstance(scalars, torch.Tensor):
        if scalars.numel() == 1:
            value = scalars.item()
            # Keep the Python type of the original value in ``values``: an
            # ``int`` scalar promotes an integral tensor differently from a
            # ``float``.  The by-value argument is always the float the device
            # buffer would have held.
            if isinstance(value, numbers.Real):
                return [value] * len(tensors), float(value), type(value)
            return [value] * len(tensors), None, None
        _check_same_length(tensors, scalars, "scalar list")
        values = scalars.detach().cpu().tolist()
        return values, _common_scalar(values), None
    if isinstance(scalars, (list, tuple)):
        _check_same_length(tensors, scalars, "scalar list")
        values = list(scalars)
        return values, _common_scalar(values), None
    if isinstance(scalars, numbers.Real):
        return [scalars] * len(tensors), float(scalars), type(scalars)
    return [scalars] * len(tensors), None, None


def _finish_inplace(
    results: List[torch.Tensor],
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for dst_view, staged in writebacks:
        dst_view.copy_(staged)
        _STATS.writebacks += 1
    del results


def foreach_binary_list(
    self: Sequence[torch.Tensor],
    other: Sequence[torch.Tensor],
    fn: Callable,
    *,
    inplace: bool = False,
    alpha: float = 1,
    allowed_dtypes: Optional[frozenset] = None,
    dtype_error: Optional[str] = None,
    dtype_error_type: type = RuntimeError,
    out_dtype_fn: Optional[Callable[[torch.dtype], torch.dtype]] = None,
    out_dtypes: Optional[Sequence[torch.dtype]] = None,
    copy_semantics: bool = False,
    scalar_operand: bool = False,
) -> List[torch.Tensor]:
    """``self[i] <op> alpha * other[i]`` for every position, one launch per group.

    Both lists are paired position by position, so they must have equal length;
    ATen raises rather than broadcasting across the list.

    ``alpha`` scales the second operand inside the kernel.

    ``copy_semantics`` replaces type promotion with assignment semantics:
    ``_foreach_copy`` keeps the destination dtype and simply casts the source
    into it, in both the functional and the in-place form.

    ``out_dtypes`` lets the caller pin the result dtype per position. The
    ``.Tensor`` overloads need it: their operand is a scalar tensor expanded to
    the list's shape, and promoting against the *expanded* view would widen an
    fp16 list to fp32, where ATen keeps fp16 because a zero-dimensional tensor
    counts as a scalar.

    ``scalar_operand`` marks the second list as one repeated single-element
    tensor (the ``.Tensor`` broadcast); the kernel then reads it as a uniform
    scalar instead of indexing it element by element.
    """
    self = check_tensor_list(self)
    other = check_tensor_list(other)
    _check_same_length(self, other, "tensor list")
    if out_dtypes is not None:
        _check_same_length(self, out_dtypes, "output dtype list")
    _STATS.reset()

    results: List[torch.Tensor] = []
    buckets: Dict[Tuple[Any, torch.dtype, torch.dtype], Tuple[List, List, List]] = {}
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    prev_key = None
    prev_promotion = None
    prev_out_dtype = None
    for i, (a, b) in enumerate(zip(self, other)):
        if allowed_dtypes is not None and a.dtype not in allowed_dtypes:
            check_dtype_supported(
                a.dtype, allowed_dtypes, dtype_error, dtype_error_type
            )
        if copy_semantics:
            # The destination dtype is the source of truth; the operand is cast
            # into it rather than promoted against it.
            out_dtype = a.dtype
        elif out_dtypes is not None:
            out_dtype = out_dtypes[i]
        else:
            # Two operands of the *same* dtype fold to that dtype whatever their
            # ranks are -- the weak-scalar rule only reorders categories -- so
            # the common same-dtype list skips both the probe and the four
            # ``dim()`` calls that key it.
            a_dtype = a.dtype
            if a_dtype is b.dtype:
                promotion = a_dtype
            else:
                promotion = (a_dtype, a.dim() == 0, b.dtype, b.dim() == 0)
            if promotion != prev_promotion:
                out_dtype = a_dtype if a_dtype is b.dtype else _promote_dtypes(a, b)
                if out_dtype_fn is not None:
                    out_dtype = out_dtype_fn(out_dtype)
                prev_promotion = promotion
                prev_out_dtype = out_dtype
            else:
                out_dtype = prev_out_dtype
        contiguous_a = a.is_contiguous()
        src_a = a if contiguous_a else _dense(a)
        src_b = b if b.is_contiguous() else _dense(b)
        if inplace:
            if not copy_semantics and out_dtype != a.dtype:
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(a.dtype).replace('torch.', '').capitalize()}"
                )
            if not contiguous_a and has_internal_overlap(a):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            dst = src_a
            if dst is not a:
                writebacks.append((a, dst))
            results.append(a)
        else:
            # The one-argument form skips the keyword/dtype dispatch that the
            # explicit form pays on every one of the sixteen allocations.
            dst = (
                torch.empty_like(src_a)
                if out_dtype is src_a.dtype
                else torch.empty_like(src_a, dtype=out_dtype)
            )
            results.append(dst)
        key = (a.device, src_a.dtype, src_b.dtype)
        if key != prev_key:
            bucket = buckets.get(key)
            if bucket is None:
                bucket = buckets[key] = ([], [], [])
            prev_key = key
        bucket[0].append(src_a)
        bucket[1].append(src_b)
        bucket[2].append(dst)

    for (device, a_dt, b_dt), (a_list, b_list, d_list) in buckets.items():
        _launch_binary_group(
            a_list, b_list, d_list, fn, a_dt, b_dt, device, alpha, scalar_operand
        )

    _finish_inplace(results, writebacks)
    return results


def _launch_binary_group(
    a_list: List[torch.Tensor],
    b_list: List[torch.Tensor],
    d_list: List[torch.Tensor],
    fn: Callable,
    a_dt: torch.dtype,
    b_dt: torch.dtype,
    device: Any,
    alpha: float = 1,
    b_scalar: bool = False,
) -> None:
    numels = [t.numel() for t in a_list]
    max_numel = max(numels)
    if max_numel == 0:
        return
    block = _pick_block(numels)
    nt = len(a_list)
    meta = _device_meta(
        [t.data_ptr() for t in a_list]
        + [t.data_ptr() for t in b_list]
        + [t.data_ptr() for t in d_list]
        + numels,
        device,
    )
    grid = (nt, triton.cdiv(max_numel, block))
    foreach_binary_kernel[grid](
        meta,
        fn,
        nt,
        tl_dtype(a_dt),
        tl_dtype(b_dt),
        tl_dtype(d_list[0].dtype),
        alpha,
        b_scalar,
        block,
    )
    _STATS.executor_launches += 1
    _STATS.groups += 1


def foreach_binary_scalar(
    self: Sequence[torch.Tensor],
    scalars: Any,
    fn: Callable,
    *,
    inplace: bool = False,
    alpha: float = 1,
    allowed_dtypes: Optional[frozenset] = None,
    dtype_error: Optional[str] = None,
    dtype_error_type: type = RuntimeError,
    out_dtype_fn: Optional[Callable[[torch.dtype], torch.dtype]] = None,
) -> List[torch.Tensor]:
    """``self[i] <op> alpha * scalar`` -- serves the ``.Scalar`` / ``.ScalarList`` forms.

    ``scalars`` is either one number (broadcast to every tensor) or a sequence
    of one number per tensor.  ``alpha`` scales the scalar inside the kernel.
    """
    self = check_tensor_list(self)
    values, common, vclass = _resolve_scalars(self, scalars)
    _STATS.reset()

    results: List[torch.Tensor] = []
    buckets: Dict[Tuple[Any, torch.dtype, torch.dtype], Tuple[List, List, List]] = {}
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    prev_key = None
    prev_promotion = None
    prev_out_dtype = None
    for t, v in zip(self, values):
        if allowed_dtypes is not None and t.dtype not in allowed_dtypes:
            check_dtype_supported(
                t.dtype, allowed_dtypes, dtype_error, dtype_error_type
            )
        # ``_promote_with_scalar`` memoises on exactly this triple, so a run of
        # tensors sharing it can reuse the previous answer instead of paying
        # for another lookup.  The broadcast form makes the value's class the
        # same object for every position, so only the dtype is looked at.
        promotion = (
            (t.dtype, vclass)
            if vclass is not None
            else (
                t.dtype,
                type(v),
                isinstance(v, torch.Tensor) and v.dim() == 0,
            )
        )
        if promotion != prev_promotion:
            out_dtype = _promote_with_scalar(t.dtype, v)
            if out_dtype_fn is not None:
                out_dtype = out_dtype_fn(out_dtype)
            prev_promotion = promotion
            prev_out_dtype = out_dtype
        else:
            out_dtype = prev_out_dtype
        contiguous = t.is_contiguous()
        src = t if contiguous else _dense(t)
        if inplace:
            if out_dtype != t.dtype:
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(t.dtype).replace('torch.', '').capitalize()}"
                )
            if not contiguous and has_internal_overlap(t):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            dst = src
            if dst is not t:
                writebacks.append((t, dst))
            results.append(t)
        else:
            # Same one-argument shortcut as the list path: the explicit dtype
            # form pays for keyword parsing on each of the sixteen allocations.
            dst = (
                torch.empty_like(src)
                if out_dtype is src.dtype
                else torch.empty_like(src, dtype=out_dtype)
            )
            results.append(dst)
        key = (t.device, src.dtype, out_dtype)
        if key != prev_key:
            bucket = buckets.get(key)
            if bucket is None:
                bucket = buckets[key] = ([], [], [])
            prev_key = key
        bucket[0].append(src)
        bucket[1].append(dst)
        bucket[2].append(v)

    for (device, in_dt, out_dt), (srcs, dsts, vals) in buckets.items():
        _launch_scalar_group(
            srcs, dsts, vals, fn, in_dt, out_dt, device, alpha, common
        )

    _finish_inplace(results, writebacks)
    return results


def _launch_scalar_group(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    vals: List[Any],
    fn: Callable,
    in_dt: torch.dtype,
    out_dt: torch.dtype,
    device: Any,
    alpha: float = 1,
    common: Optional[float] = None,
) -> None:
    numels = [t.numel() for t in srcs]
    max_numel = max(numels)
    if max_numel == 0:
        return
    block = _pick_block(numels)
    nt = len(srcs)
    meta = _device_meta(
        [t.data_ptr() for t in srcs] + [t.data_ptr() for t in dsts] + numels,
        device,
    )
    # The scalars ride in a device array of the *output* dtype, so integer
    # operators keep integer semantics instead of going through a float.
    #
    # The dtype is settled on the host before the transfer rather than by casting
    # the device tensor afterwards: that extra ``.to(dtype)`` costs a kernel and
    # an allocation, and measured 0.32ms per call -- on its own more than
    # PyTorch's entire foreach call.
    s_dtype = out_dt if out_dt.is_floating_point else torch.float32
    # ``.Scalar`` broadcasts one value into every slot, so when the values are
    # all equal there is nothing for an array to index: the value travels as a
    # kernel argument instead.  That drops a host->device copy and a device
    # allocation (~9us) from every such call.  ``common`` is already known for
    # the broadcast forms; only a real list needs the equality scan.
    if common is None:
        common = _common_scalar(vals)
    by_value = common is not None
    if by_value:
        scalar_buf = meta  # unused by the kernel; keeps the pointer slot valid
        scalar_value = common
    else:
        scalar_buf = torch.tensor([float(v) for v in vals], dtype=s_dtype).to(
            device, non_blocking=True
        )
        scalar_value = 0.0
    grid = (nt, triton.cdiv(max_numel, block))
    foreach_binary_scalar_kernel[grid](
        meta,
        scalar_buf,
        scalar_value,
        fn,
        nt,
        tl_dtype(in_dt),
        tl_dtype(out_dt),
        tl_dtype(s_dtype),
        alpha,
        by_value,
        block,
    )
    _STATS.executor_launches += 1
    _STATS.groups += 1


def foreach_ternary(
    self: Sequence[torch.Tensor],
    tensor1: Sequence[torch.Tensor],
    tensor2: Sequence[torch.Tensor],
    scalars: Any,
    fn: Callable,
    *,
    inplace: bool = False,
    allowed_dtypes: Optional[frozenset] = None,
) -> List[torch.Tensor]:
    """``fn(self[i], tensor1[i], tensor2[i], scalar[i])`` for every position.

    Serves ``addcmul`` / ``addcdiv``, whose value argument is either one number
    or one per tensor.
    """
    self = check_tensor_list(self)
    tensor1 = check_tensor_list(tensor1)
    tensor2 = check_tensor_list(tensor2)
    _check_same_length(self, tensor1, "tensor list")
    _check_same_length(self, tensor2, "tensor list")
    values, common, vclass = _resolve_scalars(self, scalars)
    _STATS.reset()

    # Homogeneous fast path: one floating dtype, one device, all dense.  The
    # general loop below still handles mixed dtypes (each promotion is folded
    # once and reused), views (``_dense``), and disallowed dtypes (which raise
    # from ``check_dtype_supported`` inside the loop).
    homogeneous = _ternary_homogeneous(self, tensor1, tensor2, allowed_dtypes)
    if homogeneous is not None:
        device = self[0].device
        dsts = self if inplace else [torch.empty_like(t) for t in self]
        _launch_ternary_group(
            self,
            tensor1,
            tensor2,
            dsts,
            values,
            fn,
            homogeneous,
            homogeneous,
            device,
            common,
        )
        return list(self) if inplace else dsts

    results: List[torch.Tensor] = []
    buckets: Dict[Tuple[Any, torch.dtype, torch.dtype], Tuple[List, ...]] = {}
    writebacks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    prev_key = None
    prev_promotion = None
    prev_out_dtype = None
    for a, b, c, v in zip(self, tensor1, tensor2, values):
        if allowed_dtypes is not None and a.dtype not in allowed_dtypes:
            check_dtype_supported(a.dtype, allowed_dtypes)
        # ``result_type`` pairs two tensors or a tensor and a number, so the
        # three-way promotion is folded through a zero-element probe of the
        # intermediate dtype rather than by chaining dtypes directly.  The
        # integer fallback below also looks at the value's type, so that joins
        # the memo key; the tuple is the same one ``_promote_dtypes`` and
        # ``_promote_with_scalar`` already key on.
        #
        # Three operands of the *same* dtype fold to that dtype whatever their
        # ranks are, so the common case needs neither the probe nor the six
        # ``dim()`` calls that key it.
        if vclass is not None:
            vkey = vclass
        else:
            vkey = type(v)
            if vkey is torch.Tensor:
                vkey = (torch.Tensor, v.dim() == 0)
        a_dtype = a.dtype
        same = a_dtype is b.dtype and a_dtype is c.dtype
        if same:
            promotion = (a_dtype, vkey)
        else:
            promotion = (
                a_dtype,
                a.dim() == 0,
                b.dtype,
                b.dim() == 0,
                c.dtype,
                c.dim() == 0,
                vkey,
            )
        if promotion != prev_promotion:
            out_dtype = a_dtype if same else _promote_dtypes(a, b, c)
            if not out_dtype.is_floating_point and not out_dtype.is_complex:
                out_dtype = _promote_with_scalar(a_dtype, v)
            prev_promotion = promotion
            prev_out_dtype = out_dtype
        else:
            out_dtype = prev_out_dtype
        contiguous_a = a.is_contiguous()
        src_a = a if contiguous_a else _dense(a)
        src_b = b if b.is_contiguous() else _dense(b)
        src_c = c if c.is_contiguous() else _dense(c)
        if inplace:
            if out_dtype != a.dtype:
                raise RuntimeError(
                    f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                    " can't be cast to the desired output type "
                    f"{str(a.dtype).replace('torch.', '').capitalize()}"
                )
            if not contiguous_a and has_internal_overlap(a):
                raise RuntimeError(
                    "unsupported operation: more than one element of the "
                    "written-to tensor refers to a single memory location. "
                    "Please clone() the tensor before performing the operation."
                )
            dst = src_a
            if dst is not a:
                writebacks.append((a, dst))
            results.append(a)
        else:
            # The one-argument form skips the keyword/dtype dispatch that the
            # explicit form pays on every one of the sixteen allocations.
            dst = (
                torch.empty_like(src_a)
                if out_dtype is src_a.dtype
                else torch.empty_like(src_a, dtype=out_dtype)
            )
            results.append(dst)
        key = (a.device, src_a.dtype, out_dtype)
        if key != prev_key:
            bucket = buckets.get(key)
            if bucket is None:
                bucket = buckets[key] = ([], [], [], [], [])
            prev_key = key
        bucket[0].append(src_a)
        bucket[1].append(src_b)
        bucket[2].append(src_c)
        bucket[3].append(dst)
        bucket[4].append(v)

    for (device, in_dt, out_dt), (a_l, b_l, c_l, d_l, v_l) in buckets.items():
        _launch_ternary_group(
            a_l, b_l, c_l, d_l, v_l, fn, in_dt, out_dt, device, common
        )

    _finish_inplace(results, writebacks)
    return results


_TERNARY_UPLOAD_TYPES = {
    torch.float32: ("f", torch.float32),
    torch.float64: ("d", torch.float64),
}


def _ternary_upload(values: List[Any], dtype: torch.dtype, device: Any) -> torch.Tensor:
    """Upload one scalar per tensor to ``device`` as a ``len(values)`` buffer.

    ``torch.tensor(values).to(device)`` pays the Python ``Tensor.to`` /
    ``copy_`` registrations on top of the transfer.  Building the host side
    with ``array`` + ``frombuffer`` keeps it out of the dispatcher, and
    redispatching ``aten::copy_`` at ``CompositeExplicitAutograd`` skips the
    Python registration.  The host buffer stays pageable and freshly built,
    so the staging copy is ordered before the caller can reuse it.
    """
    spec = _TERNARY_UPLOAD_TYPES.get(dtype)
    if spec is None or _ATEN_COPY_REDISPATCH is None or _ATEN_COMPOSITE_KEYSET is None:
        # fp16/bf16 have no ``array`` typecode; the generic path is the only
        # way to build a host tensor of that dtype.
        return torch.tensor([float(v) for v in values], dtype=dtype).to(
            device, non_blocking=True
        )
    code, host_dtype = spec
    host = torch.frombuffer(
        array.array(code, [float(v) for v in values]), dtype=host_dtype
    )
    out = torch.empty(len(values), dtype=host_dtype, device=device)
    _ATEN_COPY_REDISPATCH(_ATEN_COMPOSITE_KEYSET, out, host, True)
    return out


def _ternary_homogeneous(
    a_l: Sequence[torch.Tensor],
    b_l: Sequence[torch.Tensor],
    c_l: Sequence[torch.Tensor],
    allowed_dtypes: Optional[frozenset],
) -> Optional[torch.dtype]:
    """The shared dtype when every operand agrees, else ``None``.

    Three operands of one floating dtype on one device, all dense, need no
    promotion, no densifying and no bucketing: the whole per-tensor
    bookkeeping in :func:`foreach_ternary` is then pure overhead.  ``get_device``
    returns an int (``-1`` on CPU) and measured ~1.5us cheaper over 48 tensors
    than building ``torch.device`` objects to compare.
    """
    dtype = a_l[0].dtype
    if not dtype.is_floating_point:
        return None
    if allowed_dtypes is not None and dtype not in allowed_dtypes:
        return None
    dev = a_l[0].get_device()
    for a, b, c in zip(a_l, b_l, c_l):
        if a.dtype is not dtype or b.dtype is not dtype or c.dtype is not dtype:
            return None
        if a.get_device() != dev or b.get_device() != dev or c.get_device() != dev:
            return None
        if not (a.is_contiguous() and b.is_contiguous() and c.is_contiguous()):
            return None
    return dtype


def _launch_ternary_group(
    a_l: List[torch.Tensor],
    b_l: List[torch.Tensor],
    c_l: List[torch.Tensor],
    d_l: List[torch.Tensor],
    v_l: List[Any],
    fn: Callable,
    in_dt: torch.dtype,
    out_dt: torch.dtype,
    device: Any,
    common: Optional[float] = None,
) -> None:
    numels = [t.numel() for t in a_l]
    max_numel = max(numels)
    if max_numel == 0:
        return
    block = _pick_block(numels)
    nt = len(a_l)
    meta = _device_meta(
        [t.data_ptr() for t in a_l]
        + [t.data_ptr() for t in b_l]
        + [t.data_ptr() for t in c_l]
        + [t.data_ptr() for t in d_l]
        + numels,
        device,
    )
    s_dtype = out_dt if out_dt.is_floating_point else torch.float32
    # Same treatment as the binary scalar path: the ``.Scalar`` form supplies one
    # value for the whole list, so it travels as a kernel argument instead of a
    # device buffer.  The ``.ScalarList`` form still needs the array.
    if common is None:
        common = _common_scalar(v_l)
    by_value = common is not None
    if by_value:
        scalar_buf = meta  # unused by the kernel; keeps the pointer slot valid
        scalar_value = common
    else:
        scalar_buf = _ternary_upload(v_l, s_dtype, device)
        scalar_value = 0.0
    grid = (nt, triton.cdiv(max_numel, block))
    foreach_ternary_kernel[grid](
        meta,
        scalar_buf,
        scalar_value,
        fn,
        nt,
        tl_dtype(in_dt),
        tl_dtype(out_dt),
        tl_dtype(s_dtype),
        by_value,
        block,
    )
    _STATS.executor_launches += 1
    _STATS.groups += 1
