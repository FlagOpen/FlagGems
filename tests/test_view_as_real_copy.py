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
from . import conftest as cfg

# view_as_real_copy consumes a complex tensor and produces the real tensor one
# size down: complex32 -> float16, complex64 -> float32, complex128 -> float64.
# The tested dtypes are therefore the complex counterparts of the shared float
# dtype set: `utils.FLOAT_DTYPES` supplies the precision choices and the
# float64 entry (complex128) is appended explicitly because that shared set
# stops at float32, subject to fp64 support. Every non-complex dtype (including
# all real dtypes) is rejected, so no real dtype is ever tested.
_FLOAT_TO_COMPLEX = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}
VIEW_AS_REAL_COMPLEX_DTYPES = [
    _FLOAT_TO_COMPLEX[d] for d in utils.FLOAT_DTYPES if d in _FLOAT_TO_COMPLEX
]
if not VIEW_AS_REAL_COMPLEX_DTYPES:
    # QUICK_MODE keeps only bfloat16 in the shared set, which has no complex
    # form; fall back to the two universally supported complex types.
    VIEW_AS_REAL_COMPLEX_DTYPES = [torch.complex32, torch.complex64]
if not cfg.QUICK_MODE and utils.fp64_is_supported:
    VIEW_AS_REAL_COMPLEX_DTYPES.append(torch.complex128)

if cfg.QUICK_MODE:
    # (input_shape,) tuples of complex tensors.
    VIEW_AS_REAL_SHAPES = [
        (4,),
        (2, 5),
    ]
else:
    # (input_shape,) tuples; covers 0-dim (-> real (2,)), 1-D/2-D/3-D/5-D
    # inputs and empty tensors.
    VIEW_AS_REAL_SHAPES = [
        (),
        (8,),
        (4, 6),
        (10, 8),
        (2, 5, 7),
        (2, 3, 4),
        (2, 2, 2, 3),
        (0,),
        (3, 0),
        (1,),
    ]


def _as_float32(t):
    # complex32/float16 outputs have no full-math CPU support; a lossless
    # float32 upcast keeps the value comparison exact everywhere.
    if t.dtype == torch.float16:
        return t.to(torch.float32)
    return t


def _assert_matches(res, ref, inp):
    assert res.dtype == ref.dtype, f"dtype {res.dtype} != {ref.dtype}"
    assert tuple(res.shape) == tuple(ref.shape)
    assert res.is_contiguous()
    assert not torch._C._is_alias_of(res, inp)
    utils.gems_assert_equal(_as_float32(res), _as_float32(ref))


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("input_shape", VIEW_AS_REAL_SHAPES)
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy(input_shape, dtype):
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)

    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_dtype_mapping(dtype):
    inp = torch.randn(6, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.view_as_real_copy(inp)

    expected = {
        torch.complex32: torch.float16,
        torch.complex64: torch.float32,
        torch.complex128: torch.float64,
    }[dtype]
    assert res_out.dtype == expected
    assert res_out.shape == (6, 2)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_values(dtype):
    # The reinterpret must be bit-exact: last dim index 0 = real part, index 1
    # = imag part.
    inp = torch.randn(5, 4, dtype=dtype, device=flag_gems.device)
    res = flag_gems.view_as_real_copy(inp)
    assert torch.equal(res[..., 0].to(torch.float32), inp.real.to(torch.float32))
    assert torch.equal(res[..., 1].to(torch.float32), inp.imag.to(torch.float32))


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_noncontiguous(dtype):
    # Every stride is legal for view_as_real; the strided gather path must
    # honor the real-view strides.
    pool = torch.randn(8, 6, dtype=dtype, device=flag_gems.device)
    inp = pool[::2]
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_transposed(dtype):
    base = torch.randn(3, 5, dtype=dtype, device=flag_gems.device)
    inp = base.transpose(0, 1)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_expanded(dtype):
    # Zero-stride batch dims must be materialized into concrete copies.
    base = torch.randn(1, dtype=dtype, device=flag_gems.device)
    inp = base.expand(4)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_storage_offset(dtype):
    base = torch.randn(20, dtype=dtype, device=flag_gems.device)
    inp = base[4:12]
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_1d_strided(dtype):
    # A 1-D strided complex view has complex stride != 1; the real view then
    # exposes a non-contiguous layout the kernel must decompose.
    base = torch.randn(20, dtype=dtype, device=flag_gems.device)
    inp = base[::2]
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_independent_storage(dtype):
    inp = torch.randn(8, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.view_as_real_copy(inp)
    ref = utils.to_reference(res_out.clone())

    inp.add_(1.0)
    # The copy must not change when the source is mutated in place.
    utils.gems_assert_equal(_as_float32(res_out), _as_float32(ref))


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COMPLEX_DTYPES)
def test_accuracy_view_as_real_copy_conj(dtype):
    # A lazy conjugate input must be resolved before the reinterpret.
    inp = torch.randn(4, 3, dtype=dtype, device=flag_gems.device)
    conj_inp = inp.conj()
    ref_inp = utils.to_reference(conj_inp)

    ref_out = torch.ops.aten.view_as_real_copy(ref_inp)
    res_out = flag_gems.view_as_real_copy(conj_inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_real_copy
def test_accuracy_view_as_real_copy_empty():
    inp = torch.randn(0, dtype=torch.complex64, device=flag_gems.device)
    res_out = flag_gems.view_as_real_copy(inp)
    assert res_out.shape == (0, 2)
    assert res_out.dtype == torch.float32

    inp2 = torch.randn(3, 0, dtype=torch.complex64, device=flag_gems.device)
    res2 = flag_gems.view_as_real_copy(inp2)
    assert res2.shape == (3, 0, 2)
    ref2 = torch.ops.aten.view_as_real_copy(utils.to_reference(inp2))
    _assert_matches(res2, ref2, inp2)


@pytest.mark.view_as_real_copy
def test_accuracy_view_as_real_copy_zero_dim():
    inp = torch.randn((), dtype=torch.complex64, device=flag_gems.device)
    res = flag_gems.view_as_real_copy(inp)
    assert res.shape == (2,)
    ref = torch.ops.aten.view_as_real_copy(utils.to_reference(inp))
    _assert_matches(res, ref, inp)


@pytest.mark.view_as_real_copy
def test_accuracy_view_as_real_copy_error_paths():
    dev = flag_gems.device

    # Non-complex dtypes are rejected, matching PyTorch.
    for bad_dtype in (torch.float32, torch.float16, torch.int32, torch.bool):
        with pytest.raises(RuntimeError):
            torch.ops.aten.view_as_real_copy(
                torch.zeros(4, dtype=bad_dtype, device=dev)
            )
        with pytest.raises(RuntimeError):
            flag_gems.view_as_real_copy(torch.zeros(4, dtype=bad_dtype, device=dev))
