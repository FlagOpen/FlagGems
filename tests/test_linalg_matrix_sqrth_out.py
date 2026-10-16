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
from .conftest import QUICK_MODE

pytestmark = pytest.mark.skipif(
    not hasattr(torch.ops.aten, "linalg_matrix_sqrth"),
    reason=(
        "linalg_matrix_sqrth requires PyTorch nightly 2026-06-27 or newer, "
        "or PyTorch 2.15+"
    ),
)

DTYPES = [torch.float32, torch.complex64]
if not QUICK_MODE:
    DTYPES += [torch.float64, torch.complex128]


def _make_hpd(shape, dtype, device):
    torch.manual_seed(0)
    matrix = torch.randn(shape, dtype=dtype, device=device)
    n = shape[-1]
    return matrix @ matrix.mH + 0.5 * torch.eye(n, dtype=dtype, device=device)


def _reference(inp):
    return torch.ops.aten.linalg_matrix_sqrth(utils.to_reference(inp, True))


@pytest.mark.linalg_matrix_sqrth_out
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(4, 4), (2, 8, 8)])
def test_linalg_matrix_sqrth_out_noncontiguous(dtype, shape):
    inp = _make_hpd(shape, dtype, flag_gems.device)
    reference = _reference(inp)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device).mH
    assert not out.is_contiguous()
    pointer = out.data_ptr()
    stride = out.stride()

    with flag_gems.use_gems():
        returned = torch.ops.aten.linalg_matrix_sqrth.out(inp, out=out)

    assert returned is out
    assert returned.data_ptr() == pointer
    assert returned.stride() == stride
    utils.gems_assert_close(out, reference, dtype, reduce_dim=shape[-1])


@pytest.mark.linalg_matrix_sqrth_out
def test_linalg_matrix_sqrth_out_resize():
    inp = _make_hpd((4, 4), torch.float32, flag_gems.device)
    reference = _reference(inp)
    out = torch.empty(0, dtype=inp.dtype, device=inp.device)

    with flag_gems.use_gems():
        returned = torch.ops.aten.linalg_matrix_sqrth.out(inp, out=out)

    assert returned is out
    assert out.shape == inp.shape
    assert out.dtype == inp.dtype
    utils.gems_assert_close(out, reference, inp.dtype, reduce_dim=4)


@pytest.mark.linalg_matrix_sqrth_out
def test_linalg_matrix_sqrth_out_rejects_wrong_dtype():
    inp = _make_hpd((4, 4), torch.float32, flag_gems.device)
    out = torch.empty((4, 4), dtype=torch.float64, device=inp.device)

    with flag_gems.use_gems(), pytest.raises(RuntimeError, match="dtype"):
        torch.ops.aten.linalg_matrix_sqrth.out(inp, out=out)


@pytest.mark.linalg_matrix_sqrth_out
def test_linalg_matrix_sqrth_out_rejects_autograd():
    inp = _make_hpd((4, 4), torch.float32, flag_gems.device)
    inp.requires_grad_(True)
    out = torch.empty_like(inp)

    with flag_gems.use_gems(), pytest.raises(RuntimeError, match="out="):
        torch.ops.aten.linalg_matrix_sqrth.out(inp, out=out)
