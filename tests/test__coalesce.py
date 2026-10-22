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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Select every overload and contract case with the operator marker.
pytestmark = pytest.mark.underscore_coalesce

# Merge duplicate COO coordinates and sum their stored values.
# FP8 inputs are exercised by the rejection test.
_COALESCE_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]

# (shape, nnz); small coordinate spaces exercise duplicate reduction.
_COALESCE_CASES = tu.selected_cases(
    [
        ((4, 4), 20),
        ((5, 5), 30),
        ((8, 8), 80),
        ((16, 16), 300),
        ((64,), 200),
        ((2, 3, 4), 28),
        ((3, 5, 7), 120),
        ((4, 8, 16), 600),
        ((16, 7, 57), 2000),
        ((4, 4, 4, 4), 400),
    ],
    quick=[((4, 4), 20), ((3, 5, 7), 120)],
)

_VALUE_RANGE_CASES = tu.selected_cases(
    [((64,), 200), ((3, 5, 7), 120), ((4, 4, 4, 4), 400)], quick=[((3, 5, 7), 120)]
)

# Preserve the same-sign, non-extreme fp16/bf16 sweep: CPU/CUDA accumulation
# may differ on cancellation and overflow. Integer extremes remain covered.
_NARROW_FLOAT_DTYPES = (torch.float16, torch.bfloat16)
_EXTREME_RANGES = (("0", "max"), ("min", "0"))
_VALUE_CASES = [
    (value_range, dtype, case)
    for dtype in _COALESCE_DTYPES
    if dtype != torch.bool
    for value_range in tu.selected_ranges()
    if dtype not in _NARROW_FLOAT_DTYPES
    or (value_range != ["-1", "1"] and tuple(value_range) not in _EXTREME_RANGES)
    for case in _VALUE_RANGE_CASES
]


_OP_NAME = "_coalesce"

_ASCEND_REDUCTION_UNSUPPORTED = {
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int64,
    torch.bool,
}


def _require_native_case(dtype, *, coalesced=False):
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        pytest.skip("Kunlunxin COO FP64 target storage narrows to FP32")
    if not coalesced:
        if flag_gems.vendor_name == "ascend" and dtype in _ASCEND_REDUCTION_UNSUPPORTED:
            pytest.skip("Ascend COO reduction supports FP16, BF16, FP32 and INT32")
        if flag_gems.vendor_name == "iluvatar" and dtype == torch.float64:
            pytest.skip("Iluvatar native FP64 COO reduction returns zero value bytes")


def _expected_device():
    device = torch.device(flag_gems.device)
    if device.type != "cpu" and device.index is None:
        device = torch.device(
            device.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    return device


def _check_input(inp, dtype, *, coo=True, shape=None, nnz=None):
    try:
        assert isinstance(inp, torch.Tensor)
        assert inp.dtype == dtype, "factory must preserve the requested dtype"
        assert (
            inp.device == _expected_device()
        ), "factory must preserve the target device"
        if shape is not None:
            assert tuple(inp.shape) == tuple(
                shape
            ), "factory must preserve the requested shape"
        if coo:
            assert inp.layout == torch.sparse_coo
            if nnz is not None:
                assert inp._nnz() == nnz, "factory must preserve the requested nnz"
    except Exception as error:
        raise AssertionError("Invalid input contract or metadata inspection") from error


def _input_snapshot(inp):
    # Hold the original storages and immutable scalars independently. Only
    # initialized component elements are observed, never allocator padding.
    try:
        if inp.layout == torch.sparse_coo:
            parts = (inp._indices(), inp._values())
            geometry = (
                inp.sparse_dim(),
                inp.dense_dim(),
                inp._nnz(),
                inp.is_coalesced(),
            )
        elif inp.layout == torch.sparse_csr:
            parts = (inp.crow_indices(), inp.col_indices(), inp.values())
            geometry = (inp.sparse_dim(), inp.dense_dim(), inp._nnz())
        else:
            assert inp.layout == torch.strided
            parts = (inp,)
            geometry = (tuple(inp.stride()), inp.storage_offset())
        outer = (
            type(inp),
            inp.layout,
            inp.dtype,
            inp.device,
            tuple(inp.shape),
            inp.requires_grad,
            inp._version,
            geometry,
        )
        witnesses = []
        for part in parts:
            storage = part.untyped_storage()
            metadata = (
                type(part),
                part.dtype,
                part.device,
                tuple(part.shape),
                tuple(part.stride()),
                part.storage_offset(),
                part.data_ptr(),
                storage._cdata,
                storage.data_ptr(),
                storage.nbytes(),
                part.requires_grad,
                part._version,
                part.is_conj(),
                part.is_neg(),
            )
            payload = part.detach().contiguous().view(torch.uint8).cpu().clone()
            witnesses.append((storage, metadata, payload))
        return outer, witnesses
    except Exception as error:
        raise AssertionError("Unable to inspect the input contract") from error


def _assert_readonly(inp, before):
    try:
        after = _input_snapshot(inp)
        assert before[0] == after[0], "candidate changed input metadata"
        assert len(before[1]) == len(after[1])
        for original, current in zip(before[1], after[1]):
            assert (
                original[1] == current[1]
            ), "candidate changed input storage or component metadata"
            assert torch.equal(
                original[2], current[2]
            ), "candidate changed input values or indices"
    except Exception as error:
        raise AssertionError(
            "Invalid read-only input contract or metadata inspection"
        ) from error


def _cpu_oracle(inp, *, out=None):
    """Native CPU bootstrap for the original no-grad COO test cases."""
    assert not inp.requires_grad
    if _OP_NAME == "coalesce" and inp.layout == torch.sparse_coo and inp.is_coalesced():
        return inp
    native = getattr(torch.ops.aten, _OP_NAME)
    if inp.layout != torch.sparse_coo:
        # Preserve native rejection for the original dense/CSR negative rows.
        return native(inp.cpu())
    cpu = torch.sparse_coo_tensor(
        inp._indices().cpu(),
        inp._values().cpu(),
        inp.shape,
        dtype=inp.dtype,
        is_coalesced=inp.is_coalesced(),
    )
    result = native(cpu)
    if inp.device.type != "cpu":
        result = torch.sparse_coo_tensor(
            result._indices().to(inp.device),
            result._values().to(inp.device),
            result.shape,
            dtype=result.dtype,
            is_coalesced=result.is_coalesced(),
        )
    if out is not None:
        out.copy_(result)
        return out
    return result


def _reference_coalesce(inp, *, out=None):
    if flag_gems.vendor_name == "kunlunxin":
        return _cpu_oracle(inp, out=out)
    native = getattr(torch.ops.aten, _OP_NAME)
    return native(inp) if out is None else native.out(inp, out=out)


def _call_candidate(inp, dtype, *, out=None):
    _check_input(inp, dtype, coo=False)
    before = _input_snapshot(inp)
    device, shape, layout = before[0][3], before[0][4], before[0][1]
    candidate = getattr(flag_gems, _OP_NAME)
    try:
        result = candidate(inp) if out is None else candidate(inp, out=out)
    finally:
        # Guard failures remain AssertionErrors, even inside a caller's
        # expected native RuntimeError block. Genuine API errors propagate.
        _assert_readonly(inp, before)
    try:
        assert isinstance(result, torch.Tensor)
        assert result.device == device and result.dtype == dtype
        assert result.shape == shape and result.layout == layout
    except Exception as error:
        raise AssertionError(
            "Invalid output contract or metadata inspection"
        ) from error
    return result


def _default_bounds(dtype):
    if dtype.is_floating_point or dtype == torch.bool:
        return 0.0, 1.0
    info = torch.iinfo(dtype)
    return (-5.0, 6.0) if info.min < 0 else (0.0, 6.0)


def _make_input(shape, nnz, dtype, value_range=None, low=None, high=None, seed=2026):
    # Seeded indices may repeat; default floats are non-negative to avoid cancellation.
    gen = torch.Generator("cpu").manual_seed(seed)
    if low is None or high is None:
        low, high = _default_bounds(dtype)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in shape
        ]
    )
    if value_range is not None:
        values = tu.make_input(dtype, (nnz,), value_range).cpu()
    elif dtype == torch.bool:
        values = torch.randint(0, 2, (nnz,), dtype=torch.bool, generator=gen)
    elif dtype.is_floating_point:
        # Generate in float64 so that dtype-extreme bounds (e.g. finfo.min/max)
        # are representable while sampling; casting afterwards keeps extreme
        # float64 values from turning into inf/nan during the draw.
        values = (
            torch.rand(nnz, dtype=torch.float64, generator=gen)
            * (float(high) - float(low))
            + float(low)
        ).to(dtype)
    else:
        low_i, high_i = int(low), int(high)
        if high_i <= low_i:
            values = torch.full((nnz,), low_i, dtype=torch.int64, device="cpu").to(
                dtype
            )
        else:
            values = torch.randint(
                low_i, high_i, (nnz,), dtype=torch.int64, generator=gen
            ).to(dtype)
    return torch.sparse_coo_tensor(indices, values, shape, device=flag_gems.device)


def _make_special_values(nnz, dtype, scenario):
    base = tu.make_special_input(dtype, scenario)
    return base.repeat((nnz + base.numel() - 1) // base.numel())[:nnz]


def _make_empty_out(shape, dtype, device):
    indices = torch.empty((len(shape), 0), dtype=torch.long, device=device)
    values = torch.empty((0,), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def _assert_coalesced(res_out, ref_out, dtype, *, equal_nan=False):
    assert res_out.layout == torch.sparse_coo
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_coalesced()
    # Indices are int64 and must match exactly (unique, sorted coordinates).
    utils.gems_assert_equal(res_out.indices().cpu(), ref_out.indices().cpu())
    # Values are sums of duplicates: tolerance for float, exact for int/bool.
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(
            res_out.values().cpu(), ref_out.values().cpu(), dtype, equal_nan=equal_nan
        )
    else:
        utils.gems_assert_equal(res_out.values().cpu(), ref_out.values().cpu())


@pytest.mark.underscore_coalesce
@pytest.mark.parametrize("case", _COALESCE_CASES)
@pytest.mark.parametrize("dtype", _COALESCE_DTYPES)
def test__coalesce(case, dtype):
    _require_native_case(dtype)
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    assert not inp.is_coalesced()
    _check_input(inp, dtype, shape=shape, nnz=nnz)
    ref_inp = tu.to_reference(inp)

    ref_out = _reference_coalesce(ref_inp)
    res_out = _call_candidate(inp, dtype)

    _assert_coalesced(res_out, ref_out, dtype)
    # Coalescing returns a fresh tensor and must not mutate the input.
    assert res_out is not inp
    assert not inp.is_coalesced()


@pytest.mark.underscore_coalesce
@pytest.mark.parametrize("value_range,dtype,case", _VALUE_CASES)
def test__coalesce_value_ranges(value_range, dtype, case):
    _require_native_case(dtype)
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype, value_range=value_range)
    assert not inp.is_coalesced()
    _check_input(inp, dtype, shape=shape, nnz=nnz)
    ref_inp = tu.to_reference(inp)

    ref_out = _reference_coalesce(ref_inp)
    res_out = _call_candidate(inp, dtype)

    _assert_coalesced(res_out, ref_out, dtype)
    assert res_out is not inp
    assert not inp.is_coalesced()


@pytest.mark.underscore_coalesce
@pytest.mark.parametrize("case", _COALESCE_CASES)
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_COALESCE_DTYPES)),
)
def test__coalesce_nan_inf(case, dtype, scenario):
    _require_native_case(dtype)
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    # Keep duplicate indices and vary the special-value scenario independently.
    inp = torch.sparse_coo_tensor(
        inp._indices(),
        _make_special_values(nnz, dtype, scenario),
        shape,
        device=flag_gems.device,
    )
    assert not inp.is_coalesced()
    _check_input(inp, dtype, shape=shape, nnz=nnz)
    ref_inp = tu.to_reference(inp)

    ref_out = _reference_coalesce(ref_inp)
    res_out = _call_candidate(inp, dtype)

    _assert_coalesced(res_out, ref_out, dtype, equal_nan=True)
    assert res_out is not inp
    assert not inp.is_coalesced()


@pytest.mark.coalesce_out
@pytest.mark.parametrize("case", _COALESCE_CASES)
@pytest.mark.parametrize("dtype", _COALESCE_DTYPES)
def test__coalesce_out(case, dtype):
    _require_native_case(dtype)
    shape, nnz = case
    inp = _make_input(shape, nnz, dtype)
    assert not inp.is_coalesced()
    _check_input(inp, dtype, shape=shape, nnz=nnz)
    ref_inp = tu.to_reference(inp)
    out = _make_empty_out(shape, dtype, flag_gems.device)
    _check_input(out, dtype, shape=shape, nnz=0)
    ref_out = _make_empty_out(shape, dtype, ref_inp.device)

    ref_ret = _reference_coalesce(ref_inp, out=ref_out)
    res_ret = _call_candidate(inp, dtype, out=out)

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is out
    _assert_coalesced(res_ret, ref_ret, dtype)
    assert not inp.is_coalesced()


@pytest.mark.underscore_coalesce
def test__coalesce_rejects_dense_input():
    inp = torch.randn(4, 4, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._coalesce(inp)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _call_candidate(inp, inp.dtype)


@pytest.mark.underscore_coalesce
def test__coalesce_rejects_csr_input():
    inp = torch.randn(4, 4, dtype=torch.float32, device=flag_gems.device)
    inp = inp.to_sparse_csr()
    with pytest.raises(RuntimeError):
        torch.ops.aten._coalesce(inp)
    with pytest.raises((NotImplementedError, RuntimeError, TypeError)):
        _call_candidate(inp, inp.dtype)


@pytest.mark.underscore_coalesce
def test__coalesce_rejects_fp8_input():
    indices = torch.zeros((1, 3), dtype=torch.long, device=flag_gems.device)
    values = (
        torch.zeros(3, dtype=torch.float32, device="cpu")
        .to(torch.float8_e4m3fn)
        .to(flag_gems.device)
    )
    inp = torch.sparse_coo_tensor(indices, values, (4,), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._coalesce(inp)
    with pytest.raises((RuntimeError, NotImplementedError)):
        _call_candidate(inp, inp.dtype)
