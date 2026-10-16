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

TENSORDOT_CASES = [
    # Classic dims=2 style contraction (last d of a, first d of b).
    ((3, 4, 5), (4, 5, 6), [1, 2], [0, 1]),
    # Single contracted dim -> plain matmul-like.
    ((16, 32), (32, 24), [1], [0]),
    # Reordered / non-adjacent contracted dims.
    ((3, 5, 4, 6), (6, 4, 5, 3), [2, 1, 3], [1, 2, 0]),
    # Negative dim indices.
    ((8, 7, 9), (9, 7, 5), [-1, 1], [0, 1]),
    # Larger contraction dimension.
    ((64, 128), (128, 96), [1], [0]),
    # Outer product (no contracted dims).
    ((2, 3), (4, 5), [], []),
    # Zero-sized free dimensions.
    ((0, 3, 4), (4, 5), [2], [0]),
    ((3, 4), (4, 0, 5), [1], [0]),
    # Zero-sized contracted dimensions.
    ((3, 0), (0, 5), [1], [0]),
    # Size-1 contracted dim on self -> broadcast against other (ATen contract).
    ((3, 1), (5, 6), [1], [0]),
    # Size-1 contracted dim on other -> broadcast against self.
    ((3, 5), (1, 6), [1], [0]),
    # Size-1 contracted dim on both sides.
    ((3, 1), (1, 6), [1], [0]),
    # Mixed: one contracted dim size-1 on self, another matching.
    ((2, 1, 4), (1, 3, 4), [1], [1]),
]

def _reduce_dim(shape_a, shape_b, dims_a, dims_b):
    # Effective reduction length is the broadcast contracted size (max of the
    # two per contracted axis pair), used to size the accuracy tolerance.
    na = len(shape_a)
    nb = len(shape_b)
    k = 1
    for da, db in zip(dims_a, dims_b):
        k *= max(shape_a[da % na], shape_b[db % nb])
    return max(k, 1)

def _free_size_prod(shape, dims):
    # Product of the non-contracted (free) dimension sizes; used to size the
    # gradient tolerance since each backward matmul reduces over these dims.
    ndim = len(shape)
    contracted = [d % ndim for d in dims]
    free = [d for d in range(ndim) if d not in contracted]
    p = 1
    for d in free:
        p *= shape[d]
    return max(p, 1)

def _grad_reduce_dim(this_shape, this_dims, other_shape, other_dims):
    # Reduction depth for one operand's gradient tolerance: the backward matmul
    # reduces over the other operand's free dims, and when a contracted axis was
    # size 1 on this operand it is broadcast for the matmul and then summed back,
    # adding a reduction over the broadcast (other-side) size.
    na = len(this_shape)
    nb = len(other_shape)
    r = _free_size_prod(other_shape, other_dims)
    for da, db in zip(this_dims, other_dims):
        if this_shape[da % na] == 1 and other_shape[db % nb] != 1:
            r *= other_shape[db % nb]
    return max(r, 1)

@pytest.mark.tensordot
@pytest.mark.parametrize("shape_a, shape_b, dims_a, dims_b", TENSORDOT_CASES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_tensordot(shape_a, shape_b, dims_a, dims_b, dtype):
    torch.manual_seed(0)
    a = torch.randn(shape_a, dtype=dtype, device=flag_gems.device)
    b = torch.randn(shape_b, dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)

    ref_out = torch.tensordot(ref_a, ref_b, dims=(dims_a, dims_b))
    res_out = flag_gems.tensordot(a, b, dims_a, dims_b)

    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=_reduce_dim(shape_a, shape_b, dims_a, dims_b),
    )

@pytest.mark.tensordot
def test_tensordot_invalid_dims():
    torch.manual_seed(0)
    a = torch.randn(3, 4, 5, device=flag_gems.device)
    b = torch.randn(5, 6, device=flag_gems.device)

    # Out-of-range dimension
    with pytest.raises(IndexError):
        flag_gems.tensordot(a, b, [5], [0])

    with pytest.raises(IndexError):
        flag_gems.tensordot(a, b, [-10], [0])

    # Mismatched number of contracted dims (explicit RuntimeError, not assert).
    with pytest.raises(RuntimeError, match="number of contracted dims"):
        flag_gems.tensordot(a, b, [0, 1], [0])

    # Unequal, non-1 contracted sizes must raise (size-1 broadcast is allowed).
    c = torch.randn(3, 4, device=flag_gems.device)
    d = torch.randn(5, 6, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="need to match"):
        flag_gems.tensordot(c, d, [1], [0])

@pytest.mark.tensordot
def test_tensordot_dtype_device_validation():
    torch.manual_seed(0)
    a_f32 = torch.randn(3, 4, device=flag_gems.device, dtype=torch.float32)
    b_f16 = torch.randn(4, 5, device=flag_gems.device, dtype=torch.float16)

    # Mismatched dtype
    with pytest.raises(RuntimeError, match="same dtype"):
        flag_gems.tensordot(a_f32, b_f16, [1], [0])

    # Mismatched device (if CPU is available)
    if torch.cuda.is_available():
        a_gpu = torch.randn(3, 4, device=flag_gems.device)
        b_cpu = torch.randn(4, 5, device="cpu")
        with pytest.raises(RuntimeError, match="same device"):
            flag_gems.tensordot(a_gpu, b_cpu, [1], [0])

@pytest.mark.tensordot
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_tensordot_non_contiguous(dtype):
    # Create non-contiguous tensors via transpose
    torch.manual_seed(0)
    a = torch.randn(3, 4, 5, dtype=dtype, device=flag_gems.device).transpose(0, 2)
    b = torch.randn(5, 6, 7, dtype=dtype, device=flag_gems.device).transpose(1, 2)

    assert not a.is_contiguous()
    assert not b.is_contiguous()

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)

    ref_out = torch.tensordot(ref_a, ref_b, dims=([0], [0]))
    res_out = flag_gems.tensordot(a, b, [0], [0])

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=a.shape[0])

@pytest.mark.tensordot
def test_tensordot_unsupported_dtypes():
    # Integer dtypes should raise error
    torch.manual_seed(0)
    a_int = torch.randint(0, 10, (3, 4), device=flag_gems.device, dtype=torch.int32)
    b_int = torch.randint(0, 10, (4, 5), device=flag_gems.device, dtype=torch.int32)

    with pytest.raises((RuntimeError, NotImplementedError)):
        flag_gems.tensordot(a_int, b_int, [1], [0])

    # Complex dtypes should raise error (if not supported)
    if hasattr(torch, "complex64"):
        a_complex = torch.randn(3, 4, device=flag_gems.device, dtype=torch.complex64)
        b_complex = torch.randn(4, 5, device=flag_gems.device, dtype=torch.complex64)

        with pytest.raises((RuntimeError, NotImplementedError)):
            flag_gems.tensordot(a_complex, b_complex, [1], [0])

    # FP64 should be rejected when the device does not support it
    if not utils.fp64_is_supported:
        a_f64 = torch.randn(3, 4, device=flag_gems.device, dtype=torch.float64)
        b_f64 = torch.randn(4, 5, device=flag_gems.device, dtype=torch.float64)

        with pytest.raises(RuntimeError, match="unsupported dtype"):
            flag_gems.tensordot(a_f64, b_f64, [1], [0])

GRAD_CASES = [
    ((3, 4, 5), (4, 5, 6), [1, 2], [0, 1]),
    ((16, 32), (32, 24), [1], [0]),
    ((3, 5, 4, 6), (6, 4, 5, 3), [2, 1, 3], [1, 2, 0]),
    ((8, 7, 9), (9, 7, 5), [-1, 1], [0, 1]),
    ((3, 1), (5, 6), [1], [0]),
    ((3, 5), (1, 6), [1], [0]),
    ((2, 1, 4), (1, 3, 4), [1], [1]),
]

@pytest.mark.tensordot
@pytest.mark.parametrize("shape_a, shape_b, dims_a, dims_b", GRAD_CASES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_tensordot_backward(shape_a, shape_b, dims_a, dims_b, dtype):
    torch.manual_seed(0)
    a = torch.randn(shape_a, dtype=dtype, device=flag_gems.device, requires_grad=True)
    b = torch.randn(shape_b, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)

    ref_out = torch.tensordot(ref_a, ref_b, dims=(dims_a, dims_b))
    res_out = flag_gems.tensordot(a, b, dims_a, dims_b)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, upcast=True)

    ref_grad_a, ref_grad_b = torch.autograd.grad(ref_out, (ref_a, ref_b), ref_grad)
    res_grad_a, res_grad_b = torch.autograd.grad(res_out, (a, b), out_grad)

    # Gradients that reduce over a broadcast (size-1) contracted axis are more
    # cancellation-prone in low precision, so use a slightly larger base atol.
    utils.gems_assert_close(
        res_grad_a,
        ref_grad_a,
        dtype,
        reduce_dim=_grad_reduce_dim(shape_a, dims_a, shape_b, dims_b),
        atol=1e-2,
    )
    utils.gems_assert_close(
        res_grad_b,
        ref_grad_b,
        dtype,
        reduce_dim=_grad_reduce_dim(shape_b, dims_b, shape_a, dims_a),
        atol=1e-2,
    )
