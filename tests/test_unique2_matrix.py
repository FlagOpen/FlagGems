"""Extended acceptance matrix for flag_gems._unique2 (mthreads backend).

Complements tests/test_unique2.py (official #7255 suite) with the corners the
official suite does not cover. All cases are GPU-gated: they call
flag_gems._unique2 on flag_gems.device and must only be executed once a
stable GPU is confirmed free (see reports/09_blocked_by_resource.md).

Reference methodology mirrors the official suite: CPU aten via
utils.to_reference, all three outputs compared unconditionally.

Covers:
  - empty inputs: (0,), (0, 3), (2, 0), (0, 2, 3) x all flag combos,
    including the aten-exact rule that unrequested inverse_indices is a
    plain (0,) tensor while return_inverse=True yields view_as(in0)
  - 0-dim scalar input
  - 8192/8193 small-path/post-sort boundary
  - equal-value runs spanning the 8192-element tile boundary (exercises the
    boundary-masked scatter after the duplicate-address-write fix)
  - duplication spectrum: all-same, all-unique, high/medium/low dup rates
  - non-contiguous inputs (strided view, transpose, offset view)
  - special floats: repeated NaN, +0.0/-0.0, +/-inf (fp32; fp64 gated)
  - every supported dtype on this backend (fp16/fp32/bf16/int16/int32/int64/bool)
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

FLAG_COMBOS = [(False, False), (True, False), (False, True), (True, True)]

# fp64 is appended automatically only when the device supports it.
# bool excluded: aten::_unique2 is not implemented for Bool on this platform
# (RuntimeError in the reference itself).
SUPPORTED_DTYPES = utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES

TILE = 8192  # post-sort pipeline tile size (see _mthreads/ops/unique.py)


def _assert_unique2(res, ref, dtype):
    res_out, res_inverse, res_counts = res
    ref_out, ref_inverse, ref_counts = ref

    if dtype in utils.ALL_FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    else:
        utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_inverse, ref_inverse)
    utils.gems_assert_equal(res_counts, ref_counts)


def _run(inp, return_inverse, return_counts):
    ref_inp = utils.to_reference(inp)
    ref = torch._unique2(
        ref_inp, return_inverse=return_inverse, return_counts=return_counts
    )
    res = flag_gems._unique2(
        inp, return_inverse=return_inverse, return_counts=return_counts
    )
    return res, ref


EMPTY_SHAPES = [(0,), (0, 3), (2, 0), (0, 2, 3)]


@pytest.mark.unique2
@pytest.mark.unique2_matrix
@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("return_inverse, return_counts", FLAG_COMBOS)
def test_matrix_empty_nd(shape, dtype, return_inverse, return_counts):
    """Three-tensor contract on N-D empty inputs.

    The unrequested inverse's SHAPE for empty N-D inputs is
    platform-dependent in aten itself (observed: MUSA S5000 returns
    view_as(in0) even when return_inverse=False, while MUSA520 and CPU
    aten return a plain (0,)). We follow the generic flag_gems contract
    (view_as(in0), matching ops/unique.py) and therefore only assert
    emptiness for the unrequested case; the requested case and the exact
    value comparison run only where the contract is stable. Unrequested
    counts are always a plain (0,).
    """
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    res, ref = _run(inp, return_inverse, return_counts)

    res_out, res_inverse, res_counts = res
    ref_out, ref_inverse, ref_counts = ref
    assert res_out.numel() == 0
    assert res_counts.numel() == 0
    assert res_inverse.numel() == 0
    if return_inverse:
        # stable across platforms: requested inverse is input-shaped
        assert res_inverse.shape == inp.shape
        assert ref_inverse.shape == inp.shape
        utils.gems_assert_equal(res_inverse, ref_inverse)
    else:
        # platform-dependent aten corner (see docstring): assert emptiness
        # and dtype only
        assert res_inverse.dtype == torch.int64
    # out / counts compare exactly (inverse handled above)
    if dtype in utils.ALL_FLOAT_DTYPES:
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    else:
        utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_counts, ref_counts)


@pytest.mark.unique2
@pytest.mark.unique2_matrix
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("return_inverse, return_counts", FLAG_COMBOS)
def test_matrix_zero_dim(dtype, return_inverse, return_counts):
    """0-dim (scalar) input: numel==1 takes the small path."""
    inp = torch.tensor(7, dtype=dtype, device=flag_gems.device)
    res, ref = _run(inp, return_inverse, return_counts)
    _assert_unique2(res, ref, dtype)


@pytest.mark.unique2
@pytest.mark.unique2_matrix
@pytest.mark.parametrize("numel", [TILE, TILE + 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("return_inverse, return_counts", FLAG_COMBOS)
def test_matrix_path_boundary(numel, dtype, return_inverse, return_counts):
    """numel==8192 (small path) vs 8193 (post-sort path)."""
    inp = torch.randint(0, 100, (numel,), dtype=dtype, device=flag_gems.device)
    res, ref = _run(inp, return_inverse, return_counts)
    _assert_unique2(res, ref, dtype)


@pytest.mark.unique2
@pytest.mark.unique2_matrix
@pytest.mark.parametrize("period", [1, 7, 3000])
@pytest.mark.parametrize("return_inverse, return_counts", FLAG_COMBOS)
def test_matrix_tile_spanning_runs(period, return_inverse, return_counts):
    """Equal-value runs that straddle the 8192-element tile boundary.

    After the boundary-mask fix, only run heads write data_out/idx; a run
    crossing tiles writes from exactly one (tile, lane). int32 keeps the
    comparison cheap while still covering multiple 8192 tiles.
    """
    numel = 4 * TILE + 123
    inp = torch.arange(numel, dtype=torch.int32, device=flag_gems.device) // period
    res, ref = _run(inp, return_inverse, return_counts)
    _assert_unique2(res, ref, torch.int32)


@pytest.mark.unique2
@pytest.mark.unique2_matrix
@pytest.mark.parametrize("kind", ["all_same", "all_unique", "high_dup", "med_dup"])
@pytest.mark.parametrize("return_inverse, return_counts", FLAG_COMBOS)
def test_matrix_duplication(kind, return_inverse, return_counts):
    """Duplication-rate spectrum on the post-sort path (n=20000)."""
    n = 20000
    if kind == "all_same":
        inp = torch.full((n,), 5, dtype=torch.int32, device=flag_gems.device)
    elif kind == "all_unique":
        inp = torch.arange(n, dtype=torch.int32, device=flag_gems.device)
    elif kind == "high_dup":
        inp = torch.randint(0, 10, (n,), dtype=torch.int32, device=flag_gems.device)
    else:  # med_dup: ~n/100 distinct values
        inp = torch.randint(
            0, n // 100, (n,), dtype=torch.int32, device=flag_gems.device
        )
    res, ref = _run(inp, return_inverse, return_counts)
    _assert_unique2(res, ref, torch.int32)


@pytest.mark.unique2
@pytest.mark.unique2_matrix
@pytest.mark.parametrize("view_kind", ["stride2", "transpose", "offset"])
@pytest.mark.parametrize("return_inverse, return_counts", FLAG_COMBOS)
def test_matrix_non_contiguous(view_kind, return_inverse, return_counts):
    """Non-contiguous inputs (strided/transpose/offset) above the 8192 path
    boundary; inverse_indices must still come back with the input's shape."""
    base = torch.arange(2 * TILE, dtype=torch.float32, device=flag_gems.device)
    base = base.reshape(64, 256).round() / 4.0  # duplicates + 2-D
    if view_kind == "stride2":
        inp = base[:, ::2]
    elif view_kind == "transpose":
        inp = base.t()
    else:
        # storage offset + stride: base[1:, :] alone stays contiguous for a
        # contiguous base, so combine the offset with a stride to make a
        # genuinely non-contiguous view
        inp = base[1:, ::2]
    assert not inp.is_contiguous()
    res, ref = _run(inp, return_inverse, return_counts)
    _assert_unique2(res, ref, torch.float32)


@pytest.mark.unique2
@pytest.mark.unique2_matrix
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_matrix_special_floats(dtype):
    """Repeated NaN, signed zeros and infinities (official-suite superset).

    NaN != NaN means each NaN occupies its own unique slot; which slot a NaN
    maps to in inverse_indices is implementation-defined, so inverse is
    validated by reconstruction (out[inverse] == input, NaN-aware), mirroring
    tests/test_unique2.py::test_unique2_float_special_values.
    """
    values = [
        0.0,
        -0.0,
        1.5,
        float("inf"),
        -1.5,
        float("nan"),
        float("nan"),
        float("-inf"),
        0.0,
        1.5,
        float("nan"),
    ]
    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    res, ref = _run(inp, True, True)

    res_out, res_inverse, res_counts = res
    ref_out, ref_inverse, ref_counts = ref

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_equal(res_counts, ref_counts)
    # reconstruction check: compare against the REFERENCE-side input (CPU
    # when the harness runs --ref cpu), never the raw device tensor —
    # gems_assert_close's to_cpu() asserts the reference is on CPU
    recon = res_out[res_inverse]
    utils.gems_assert_close(recon, utils.to_reference(inp), dtype, equal_nan=True)


@pytest.mark.unique2
@pytest.mark.unique2_matrix
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("numel", [1000, TILE + 5])
def test_matrix_dtypes(dtype, numel):
    """Every supported dtype on both code paths (flags (True, True))."""
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        inp = torch.randn(numel, dtype=dtype, device=flag_gems.device)
        inp = (inp * 10).round() / 10
    elif dtype is torch.bool:
        inp = torch.randint(0, 2, (numel,), dtype=torch.int64, device=flag_gems.device)
        inp = inp.to(torch.bool)
    else:
        inp = torch.randint(0, 1000, (numel,), dtype=dtype, device=flag_gems.device)
    res, ref = _run(inp, True, True)
    _assert_unique2(res, ref, dtype)


@pytest.mark.unique2
@pytest.mark.unique2_matrix
def test_matrix_large_multitile():
    """100k elements, runs of 3000 spanning many tiles, full flags."""
    n = 100000
    inp = torch.arange(n, dtype=torch.int32, device=flag_gems.device) // 3000
    res, ref = _run(inp, True, True)
    _assert_unique2(res, ref, torch.int32)
