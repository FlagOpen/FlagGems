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

# Eigenvalue decomposition needs a square matrix, and cuSOLVER only covers
# float32 among the real dtypes.
EIGVALS_SHAPES = [(2, 2), (3, 3), (5, 5), (10, 10), (20, 20)]


def _sorted_eigvals(values):
    """Order eigenvalues lexicographically by (real, imag).

    Eigenvalues come back in an unspecified order, and LAPACK and cuSOLVER
    genuinely disagree on it for some inputs, so the order has to be
    normalised before an element-wise comparison is meaningful.
    """
    ordered = values[torch.argsort(values.imag, stable=True)]
    return ordered[torch.argsort(ordered.real, stable=True)]

@pytest.mark.linalg_eigvals_out
@pytest.mark.parametrize("shape", EIGVALS_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigvals_out(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    # complex64 is the eigenvalue dtype produced for a float32 input
    ref_out = torch.empty(shape[0], dtype=torch.complex64, device=ref_inp.device)
    torch.ops.aten.linalg_eigvals.out(ref_inp, out=ref_out)

    res_out = torch.empty(shape[0], dtype=torch.complex64, device=flag_gems.device)
    returned = flag_gems.ops.linalg_eigvals_out(inp, out=res_out)

    # the out variant must write in place and return the very same tensor
    assert returned.data_ptr() == res_out.data_ptr()
    utils.gems_assert_close(
        _sorted_eigvals(res_out), _sorted_eigvals(ref_out), res_out.dtype
    )


@pytest.mark.linalg_eigvals_out
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigvals_out_resize(dtype):
    """An empty ``out`` must be resized to the eigenvalue shape."""
    torch.manual_seed(0)
    inp = torch.randn((6, 6), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty(0, dtype=torch.complex64, device=ref_inp.device)
    torch.ops.aten.linalg_eigvals.out(ref_inp, out=ref_out)

    res_out = torch.empty(0, dtype=torch.complex64, device=flag_gems.device)
    flag_gems.ops.linalg_eigvals_out(inp, out=res_out)

    assert res_out.shape == (6,)
    utils.gems_assert_close(
        _sorted_eigvals(res_out), _sorted_eigvals(ref_out), res_out.dtype
    )


@pytest.mark.linalg_eigvals_out
def test_linalg_eigvals_out_noncontiguous_storage_contract():
    torch.manual_seed(0)
    inp = torch.randn((4, 4), dtype=torch.float32, device=flag_gems.device)
    backing = torch.empty((8,), dtype=torch.complex64, device=flag_gems.device)
    out = backing[::2]
    pointer = out.untyped_storage().data_ptr()
    stride = out.stride()

    returned = flag_gems.ops.linalg_eigvals_out(inp, out=out)

    assert returned is out
    assert out.untyped_storage().data_ptr() == pointer
    assert out.stride() == stride


@pytest.mark.linalg_eigvals_out
def test_linalg_eigvals_out_rejects_real_output():
    torch.manual_seed(0)
    inp = torch.randn((4, 4), dtype=torch.float32, device=flag_gems.device)
    out = torch.empty((4,), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.ops.linalg_eigvals_out(inp, out=out)


@pytest.mark.linalg_eigvals_out
def test_linalg_eigvals_out_rejects_autograd():
    torch.manual_seed(0)
    inp = torch.randn((4, 4), dtype=torch.float32, device=flag_gems.device)
    inp.requires_grad_(True)
    out = torch.empty((4,), dtype=torch.complex64, device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises(RuntimeError, match="out="):
        torch.ops.aten.linalg_eigvals.out(inp, out=out)
