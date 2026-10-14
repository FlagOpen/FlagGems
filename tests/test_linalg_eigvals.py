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

# The registry id is ``linalg_eigvals`` (the aten name starts with an
# underscore; see naming Rule 2), so check-operator-markers expects the
# ``linalg_eigvals`` marker, while CI's marker selection (tools/test-op.sh)
# derives the marker from the implementation file stem ``_linalg_eigvals``
# via the #6359 convention, yielding ``underscore_linalg_eigvals``. Carry
# both markers so the tests are reachable from either entry point. See
# tests/test_nested_tensor_from_tensor_list.py for the same pattern.
pytestmark = [
    pytest.mark.linalg_eigvals,
    pytest.mark.underscore_linalg_eigvals,
]

# Eigenvalue decomposition needs a square matrix. The native CUDA kernel
# accepts float32/complex64/complex128 (float64 upcasts to complex128).
EIGVALS_SHAPES = [(2, 2), (3, 3), (5, 5), (10, 10), (20, 20)]


def _sorted_eigvals(values):
    """Order eigenvalues lexicographically by (real, imag).

    Eigenvalues come back in an unspecified order, and LAPACK and cuSOLVER
    genuinely disagree on it for some inputs, so the order has to be
    normalised before an element-wise comparison is meaningful.  Works on the
    flattened view so batched inputs sort the eigenvalue axis, not dim 0.
    """
    shape = values.shape
    flat = values.reshape(-1)
    ordered = flat[torch.argsort(flat.imag, stable=True)]
    ordered = ordered[torch.argsort(ordered.real, stable=True)]
    return ordered.reshape(shape)


@pytest.mark.linalg_eigvals
@pytest.mark.parametrize("shape", EIGVALS_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigvals(shape, dtype):
    """Test _linalg_eigvals accuracy against PyTorch reference."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._linalg_eigvals.default(ref_inp)
    res_out = flag_gems.ops._linalg_eigvals(inp)

    # For float32 input, the eigenvalues come back as complex64.
    utils.gems_assert_close(
        _sorted_eigvals(res_out), _sorted_eigvals(ref_out), res_out.dtype
    )


@pytest.mark.linalg_eigvals
@pytest.mark.parametrize("shape", EIGVALS_SHAPES)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linalg_eigvals_complex(shape, dtype):
    """Complex input is supported natively and must stay delegate-clean."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._linalg_eigvals.default(ref_inp)
    res_out = flag_gems.ops._linalg_eigvals(inp)

    assert res_out.dtype == dtype
    utils.gems_assert_close(
        _sorted_eigvals(res_out), _sorted_eigvals(ref_out), res_out.dtype
    )


@pytest.mark.linalg_eigvals
@pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 5, 5)])
def test_linalg_eigvals_batched(shape):
    """Batched square matrices must keep the native batched result shape."""
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._linalg_eigvals.default(ref_inp)
    res_out = flag_gems.ops._linalg_eigvals(inp)

    assert res_out.shape == ref_out.shape == (*shape[:-1],)
    utils.gems_assert_close(
        _sorted_eigvals(res_out), _sorted_eigvals(ref_out), res_out.dtype
    )
