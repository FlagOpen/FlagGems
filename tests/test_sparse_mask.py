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

import math
from contextlib import contextmanager

import pytest
import torch
from torch.autograd.function import once_differentiable

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu


def _copy_dense_physical(value, device):
    assert isinstance(value, torch.Tensor) and value.layout == torch.strided
    shape, stride, offset = tuple(value.shape), value.stride(), value.storage_offset()
    source_dtype = value.dtype
    source_conj, source_neg = value.is_conj(), value.is_neg()
    source_storage = value.untyped_storage()
    raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(
        source_storage, 0, (source_storage.nbytes(),), (1,)
    )
    # Independent storage on CPU even when the input is already CPU.
    raw = raw.to(device=device).clone()
    result = raw.view(source_dtype).as_strided(shape, stride, offset)
    if source_conj:
        result = result.conj()
    if source_neg:
        result = torch._neg_view(result)
    assert result.device == torch.device(device) and result.dtype == source_dtype
    assert tuple(result.shape) == shape and result.stride() == stride
    assert result.storage_offset() == offset
    assert result.is_conj() == source_conj and result.is_neg() == source_neg
    return result


def _copy_tensor_physical(value, device, *, preserve_requires_grad=False):
    assert isinstance(value, torch.Tensor)
    required_shape, required_dtype = tuple(value.shape), value.dtype
    if value.layout == torch.sparse_coo:
        result = torch.sparse_coo_tensor(
            _copy_dense_physical(value._indices(), device),
            _copy_dense_physical(value._values(), device),
            required_shape,
            dtype=required_dtype,
            device=device,
            is_coalesced=value.is_coalesced(),
        )
        assert result.sparse_dim() == value.sparse_dim()
        assert result.dense_dim() == value.dense_dim()
        assert result.is_coalesced() == value.is_coalesced()
        assert result._nnz() == value._nnz()
    elif value.layout == torch.strided:
        result = _copy_dense_physical(value, device)
    else:
        # Original cases use dense or COO; do not invent a compressed conversion.
        raise AssertionError(
            "CPU bootstrap transport supports original dense/COO layouts only"
        )
    assert result.device == torch.device(device)
    assert result.dtype == required_dtype and tuple(result.shape) == required_shape
    assert result.layout == value.layout
    if preserve_requires_grad and value.requires_grad:
        result.requires_grad_(True)
    return result


def _copy_tensor(value, device, *, preserve_requires_grad=False):
    # Auxiliary transport failure must not satisfy an original negative caller's
    # broad RuntimeError expectation. Actual sparse_mask/autograd calls stay out.
    try:
        return _copy_tensor_physical(
            value, device, preserve_requires_grad=preserve_requires_grad
        )
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError("sparse_mask CPU bootstrap transport failed") from error


def _assert_return_ABI(result, cpu_result, required_device):
    assert isinstance(result, torch.Tensor) and isinstance(cpu_result, torch.Tensor)
    assert result.device == required_device
    assert result.dtype == cpu_result.dtype and result.shape == cpu_result.shape
    assert result.layout == cpu_result.layout
    if result.layout == torch.sparse_coo:
        assert result.is_coalesced() == cpu_result.is_coalesced()
        assert result.sparse_dim() == cpu_result.sparse_dim()
        assert result.dense_dim() == cpu_result.dense_dim()
        assert result._nnz() == cpu_result._nnz()
        for part in (result._indices(), result._values()):
            assert part.device == required_device
        assert result._indices().dtype == cpu_result._indices().dtype
        assert result._values().dtype == cpu_result._values().dtype


class _SparseMaskCPUBridge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, mask):
        required_device = inp.device
        cpu_inp = _copy_tensor(inp, "cpu")
        cpu_mask = _copy_tensor(mask, "cpu")
        # Native CPU semantics decide dtype promotion, sparse support and errors.
        cpu_result = torch.ops.aten.sparse_mask(cpu_inp, cpu_mask)
        ctx.save_for_backward(cpu_inp, cpu_mask, mask)
        ctx.required_device = required_device
        ctx.input_dtype, ctx.input_shape, ctx.input_layout = (
            inp.dtype,
            inp.shape,
            inp.layout,
        )
        result = _copy_tensor(cpu_result, required_device)
        _assert_return_ABI(result, cpu_result, required_device)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # Accessing saved_tensors checks the original mask version, as native does.
        cpu_inp, cpu_mask, _original_mask = ctx.saved_tensors
        assert grad_output.device == ctx.required_device
        cpu_grad = _copy_tensor(grad_output, "cpu")
        # Function.forward executes with grad disabled. Recompute only genuine
        # CPU native under enable_grad to preserve its first-order gradient.
        with torch.enable_grad():
            cpu_leaf = _copy_tensor(cpu_inp, "cpu").requires_grad_(True)
            cpu_result = torch.ops.aten.sparse_mask(cpu_leaf, cpu_mask)
            (cpu_input_grad,) = torch.autograd.grad(
                cpu_result,
                cpu_leaf,
                grad_outputs=cpu_grad,
                create_graph=False,
                retain_graph=False,
            )
        result = _copy_tensor(cpu_input_grad, ctx.required_device)
        assert result.device == ctx.required_device
        assert result.dtype == ctx.input_dtype and result.shape == ctx.input_shape
        assert result.layout == ctx.input_layout
        return result, None


def sparse_mask_cpu_native(inp, mask, *, out=None):
    """NativeCPU bootstrap candidate; original argument tensors are never edited.

    Original sparse_mask tests pass bool COO masks and dense/COO self. The out
    cases have no autograd requirement; no claim about out-autograd is made.
    """
    assert isinstance(inp, torch.Tensor) and isinstance(mask, torch.Tensor)
    required_device = inp.device  # Capture original tensor device before copies.
    if mask.device != required_device:
        raise RuntimeError("self and mask devices must match before CPU transport")
    if out is None:
        return _SparseMaskCPUBridge.apply(inp, mask)
    assert isinstance(out, torch.Tensor)
    if out.device != required_device:
        raise RuntimeError("out and self devices must match before CPU transport")
    cpu_inp = _copy_tensor(inp, "cpu", preserve_requires_grad=True)
    cpu_mask = _copy_tensor(mask, "cpu", preserve_requires_grad=True)
    cpu_out = _copy_tensor(out, "cpu", preserve_requires_grad=True)
    returned = torch.ops.aten.sparse_mask.out(cpu_inp, cpu_mask, out=cpu_out)
    assert returned is cpu_out
    target = _copy_tensor(returned, required_device)
    _assert_return_ABI(target, returned, required_device)
    # Native out may overwrite/reallocate. Promise only original object identity,
    # not stable StorageImpl/data_ptr or readonly out components.
    out.data = target
    _assert_return_ABI(out, returned, required_device)
    return out


_cpu_oracle = sparse_mask_cpu_native

# Select every overload and contract case with the operator marker.
pytestmark = pytest.mark.sparse_mask

# Gather values at the sparse mask positions; both operands must have the same shape.
# The reference rejects FP8 self operands during dtype promotion.
_SPARSE_MASK_DTYPES = (
    [
        torch.int8,
        torch.uint8,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
    ]
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)

_SPARSE_MASK_STRUCT_SHAPES = [
    (16,),
    (2, 3),
    (8, 8),
    (16, 32),
    (4, 8, 16),
    (3, 7, 5, 4),
]

_SPARSE_MASK_BACKWARD_SHAPES = [(16, 32), (4, 8, 16), (3, 7, 5, 4)]

_SPARSE_MASK_NANINF_SHAPES = [(16,), (8, 8), (3, 7, 5, 4)]

# (base shape, shape after slicing the last dimension by 2).
_SPARSE_MASK_NON_CONTIGUOUS_CASES = [
    ((4, 8, 16), (4, 8, 8)),
    ((6, 10), (6, 5)),
]

# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_SPARSE_MASK_VALUE_RANGE_CASES = [
    (shape, value_range, dtype)
    for dtype in _SPARSE_MASK_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    for shape in tu.selected_shapes()
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


def _make_mask(shape, density=0.5):
    # Keep values above the threshold, then convert the bool mask to coalesced COO.
    return (torch.rand(shape, device=flag_gems.device) > density).to_sparse()


def _to_reference(inp):
    # Ascend has sparse COO tensors but lacks material native sparse_mask paths.
    # Keep the independent oracle on CPU without changing candidate inputs.
    if flag_gems.vendor_name == "ascend":
        return inp.detach().cpu().clone().requires_grad_(inp.requires_grad)
    return tu.to_reference(inp)


def _assert_equal(result, reference):
    if flag_gems.vendor_name == "ascend":
        result = result.cpu()
    tu.assert_result_equal(result, reference)


def _assert_masked(res_out, ref_out, device):
    assert res_out.device == device
    assert res_out.layout == ref_out.layout
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    _assert_equal(res_out.indices(), ref_out.indices())
    _assert_equal(res_out.values(), ref_out.values())


def _readonly_geometry(value):
    assert isinstance(value, torch.Tensor)
    state = (
        type(value),
        value.device,
        value.dtype,
        value.layout,
        tuple(value.shape),
        value.requires_grad,
        value.is_leaf,
        value._version,
        value.is_conj(),
        value.is_neg(),
    )
    if value.layout == torch.sparse_coo:
        return state + (
            value.sparse_dim(),
            value.dense_dim(),
            value._nnz(),
            value.is_coalesced(),
        )
    assert value.layout == torch.strided
    return state + (tuple(value.stride()), value.storage_offset())


def _readonly_part_state(value):
    # Keep the original StorageImpl alive, and capture scalar pointers before
    # any candidate call. A temporary COO component Python wrapper is not ABI.
    storage = value.untyped_storage()
    token = (
        int(storage._cdata),
        int(storage.data_ptr()),
        int(value.data_ptr()),
        int(storage.nbytes()),
    )
    raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(
        storage, 0, (token[3],), (1,)
    )
    return {
        "geometry": _readonly_geometry(value),
        "storage": storage,
        "token": token,
        "base": value._base,
        # Copy every original storage byte, including parent/slice padding and
        # exact NaN/signed-zero payloads. CPU input snapshots must be independent.
        "bytes": raw.cpu().clone(),
    }


def _readonly_state(value):
    geometry = _readonly_geometry(value)
    parts = (
        (value._indices(), value._values())
        if value.layout == torch.sparse_coo
        else (value,)
    )
    return {
        "value": value,
        "geometry": geometry,
        "parts": tuple(_readonly_part_state(part) for part in parts),
    }


def _assert_readonly_states(states):
    try:
        for state in states:
            value = state["value"]
            assert (
                _readonly_geometry(value) == state["geometry"]
            ), "input outer ABI/version changed"
            parts = (
                (value._indices(), value._values())
                if value.layout == torch.sparse_coo
                else (value,)
            )
            assert len(parts) == len(state["parts"])
            for part, old in zip(parts, state["parts"]):
                assert (
                    _readonly_geometry(part) == old["geometry"]
                ), "input component ABI/version changed"
                assert part._base is old["base"], "input base relation changed"
                storage = part.untyped_storage()
                token = (
                    int(storage._cdata),
                    int(storage.data_ptr()),
                    int(part.data_ptr()),
                    int(storage.nbytes()),
                )
                assert (
                    token == old["token"]
                ), "input storage identity/pointer/extent changed"
                # old['storage'] remains held throughout this check, preventing
                # recycled StorageImpl addresses from masquerading as unchanged.
                raw = torch.empty(0, dtype=torch.uint8, device=part.device).set_(
                    storage, 0, (token[3],), (1,)
                )
                assert torch.equal(
                    raw.cpu(), old["bytes"]
                ), "input physical storage bytes changed"
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError("sparse_mask readonly input observation failed") from error


@contextmanager
def _readonly_inputs(*values):
    try:
        states = tuple(_readonly_state(value) for value in values)
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError("sparse_mask readonly input snapshot failed") from error

    def check_inputs():
        _assert_readonly_states(states)

    try:
        yield check_inputs
    finally:
        # Also covers delayed autograd input mutations and positive-call errors.
        check_inputs()


def _skip_kunlunxin_native_case(inp, mask):
    if flag_gems.vendor_name != "kunlunxin" or mask._nnz() == 0:
        return
    if inp.layout == torch.sparse_coo and inp.dtype != torch.uint8 and inp._nnz() > 0:
        pytest.skip(
            "Kunlunxin native sparse_mask rejects nonempty sparse-self inputs "
            "for these dtypes"
        )
    if inp.layout == torch.strided and inp.dtype in (torch.int8, torch.int16):
        pytest.skip(
            "Kunlunxin native sparse_mask lacks int8/int16 dense-self support "
            "with a nonempty mask"
        )


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape,value_range,dtype", _SPARSE_MASK_VALUE_RANGE_CASES)
def test_sparse_mask_value_ranges(shape, value_range, dtype):
    # Use fewer mask entries for large tensors.
    numel = math.prod(shape)
    keep_threshold = 0.5 if numel <= 4096 else 0.9
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = _to_reference(inp)
    mask = _make_mask(shape, density=keep_threshold)
    _skip_kunlunxin_native_case(inp, mask)
    ref_mask = _to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    expected_device = inp.device
    with _readonly_inputs(inp, mask) as _check_inputs:
        res_out = flag_gems.sparse_mask(inp, mask)
        _check_inputs()

        _assert_masked(res_out, ref_out, expected_device)
        # The gather must not mutate either operand: the reference was computed on
        # a pristine clone, so any candidate mutation of ``inp``/``mask`` shows up
        # here.
        assert inp.device == mask.device == expected_device
        _assert_equal(inp, ref_inp)
        _assert_equal(mask, ref_mask)


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape", _SPARSE_MASK_STRUCT_SHAPES)
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask_sparse_self(shape, dtype):
    dense = tu.make_input(dtype, shape, ["-1", "1"])
    inp = dense.to_sparse()
    mask = _make_mask(shape)
    _skip_kunlunxin_native_case(inp, mask)
    ref_inp = _to_reference(inp)
    ref_mask = _to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    expected_device = inp.device
    with _readonly_inputs(inp, mask) as _check_inputs:
        res_out = flag_gems.sparse_mask(inp, mask)
        _check_inputs()

        _assert_masked(res_out, ref_out, expected_device)


@pytest.mark.sparse_mask
@pytest.mark.parametrize("base_shape,shape", _SPARSE_MASK_NON_CONTIGUOUS_CASES)
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask_non_contiguous(base_shape, shape, dtype):
    base = tu.make_input(dtype, base_shape, ["-1", "1"])
    ref_base = _to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()
    mask = _make_mask(shape)
    _skip_kunlunxin_native_case(inp, mask)
    ref_mask = _to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    expected_device = inp.device
    with _readonly_inputs(inp, mask) as _check_inputs:
        res_out = flag_gems.sparse_mask(inp, mask)
        _check_inputs()

        _assert_masked(res_out, ref_out, expected_device)


@pytest.mark.sparse_mask_out
@pytest.mark.parametrize("shape", _SPARSE_MASK_STRUCT_SHAPES)
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask_out(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    mask = _make_mask(shape)
    _skip_kunlunxin_native_case(inp, mask)
    ref_inp = _to_reference(inp)
    ref_mask = _to_reference(mask)

    out = torch.empty_like(mask, dtype=dtype)
    ref_out = torch.empty_like(ref_mask, dtype=dtype)

    torch.ops.aten.sparse_mask.out(ref_inp, ref_mask, out=ref_out)
    expected_device = inp.device
    with _readonly_inputs(inp, mask) as _check_inputs:
        res_ret = flag_gems.sparse_mask(inp, mask, out=out)
        _check_inputs()

        assert res_ret is out
        _assert_masked(res_ret, ref_out, expected_device)


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape", _SPARSE_MASK_NANINF_SHAPES)
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_SPARSE_MASK_DTYPES))
)
def test_sparse_mask_nan_inf(shape, dtype, scenario):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    specials = tu.make_special_input(dtype, scenario)
    n = min(inp.numel(), specials.numel())
    inp.flatten()[:n] = specials[:n]

    # Mask every position holding a special value (plus random extras) so the
    # nan/inf entries are guaranteed to flow into the result.
    mask_dense = torch.rand(shape, device=flag_gems.device) > 0.7
    mask_dense.flatten()[:n] = True
    mask = mask_dense.to_sparse()

    ref_inp = _to_reference(inp)
    ref_mask = _to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    expected_device = inp.device
    with _readonly_inputs(inp, mask) as _check_inputs:
        res_out = flag_gems.sparse_mask(inp, mask)
        _check_inputs()

        _assert_masked(res_out, ref_out, expected_device)


@pytest.mark.sparse_mask
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="Kunlunxin native sparse_mask backward raises invalid device function",
)
@pytest.mark.parametrize("shape", _SPARSE_MASK_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test_sparse_mask_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    mask = _make_mask(shape)
    dense_grad = tu.make_input(dtype, shape, ["-1", "1"])
    # grad_output for the sparse result: sparse COO with the mask's indices and
    # values zero outside the mask.
    grad_out = (dense_grad * mask.to_dense()).to_sparse()

    ref_inp = _to_reference(inp)
    ref_mask = _to_reference(mask)
    ref_grad_out = _to_reference(grad_out)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad_out)[0]

    expected_device = inp.device
    with _readonly_inputs(inp, mask, grad_out) as _check_inputs:
        res_out = flag_gems.sparse_mask(inp, mask)
        _check_inputs()
        _assert_masked(res_out, ref_out, expected_device)

        # The candidate must retain the reference's autograd behavior.
        assert res_out.requires_grad
        res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad_out)[0]
        _check_inputs()
        assert res_in_grad.device == expected_device
        if flag_gems.vendor_name == "ascend":
            res_in_grad = res_in_grad.cpu()
        tu.assert_result_close(res_in_grad, ref_in_grad)


@pytest.mark.sparse_mask_negative
def test_sparse_mask_shape_mismatch():
    self_t = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
    mask = _make_mask((4, 6))
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_mask(
            _to_reference(self_t),
            _to_reference(mask),
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.sparse_mask(self_t, mask)


@pytest.mark.sparse_mask_negative
def test_sparse_mask_rejects_dense_mask():
    self_t = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
    dense_mask = torch.rand(4, 5, device=flag_gems.device) > 0.5
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_mask(
            _to_reference(self_t),
            _to_reference(dense_mask),
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.sparse_mask(self_t, dense_mask)
