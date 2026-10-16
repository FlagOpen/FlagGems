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


ORMQR_SHAPES = [(32, 32), (64, 64), (128, 64), (64, 128)]
ORMQR_BATCH_SHAPES = [(2, 32, 16), (2, 3, 24, 32)]
ORMQR_DTYPES = [torch.float32, torch.float64]


def _make_inputs(shape, dtype, left):
    torch.manual_seed(0)
    m, n = shape[-2:]
    k = min(m, n)
    reflector_rows = m if left else n
    reflector = torch.randn(
        (*shape[:-2], reflector_rows, k), dtype=dtype, device=flag_gems.device
    )
    inp, tau = torch.geqrf(reflector)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return inp, tau, other


@pytest.mark.ormqr
@pytest.mark.parametrize("shape", ORMQR_SHAPES + ORMQR_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", ORMQR_DTYPES)
@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
def test_ormqr(shape, dtype, left, transpose):
    inp, tau, other = _make_inputs(shape, dtype, left)
    ref_input = utils.to_reference(inp, True)
    ref_tau = utils.to_reference(tau, True)
    ref_other = utils.to_reference(other, True)
    ref_out = torch.ormqr(
        ref_input, ref_tau, ref_other, left=left, transpose=transpose
    )

    with flag_gems.use_gems():
        result = torch.ormqr(inp, tau, other, left=left, transpose=transpose)

    assert result.shape == ref_out.shape
    assert result.dtype == ref_out.dtype
    utils.gems_assert_close(result, ref_out, dtype, reduce_dim=min(shape[-2:]))


@pytest.mark.ormqr
@pytest.mark.parametrize("left", [True, False])
def test_ormqr_rejects_invalid_reflector_dimensions(left):
    inp, tau, other = _make_inputs((8, 5), torch.float32, left)
    bad_tau = tau[..., :-1]
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.ormqr(inp, bad_tau, other, left=left)
