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


DIRICHLET_GRAD_DTYPES = [torch.float32, torch.float64]
_ALPHA_RANGES = [(0.5, 10.0), (1.0, 50.0), (0.2, 100.0)]
_SHAPES = [(2,), (8, 4), (64, 15), (20, 32)]


def _make_inputs(shape, dtype, alpha_range):
    torch.manual_seed(0)
    lo, hi = alpha_range
    alpha = torch.rand(shape, dtype=dtype, device=flag_gems.device) * (hi - lo) + lo
    total = alpha.sum(-1, keepdim=True).expand_as(alpha).contiguous()
    gamma_shape = alpha if dtype == torch.float64 else alpha.float()
    samples = torch.distributions.Gamma(
        gamma_shape, torch.ones_like(gamma_shape)
    ).sample().to(dtype)
    x = samples / samples.sum(-1, keepdim=True)
    return x, alpha, total


@pytest.mark.skipif(
    cfg.TO_CPU,
    reason="CPU ref diverges from CUDA kernel on saddle-point branch",
)
@pytest.mark.dirichlet_grad
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", DIRICHLET_GRAD_DTYPES)
@pytest.mark.parametrize("alpha_range", _ALPHA_RANGES)
def test_dirichlet_grad_regimes(shape, dtype, alpha_range):
    x, alpha, total = _make_inputs(shape, dtype, alpha_range)
    x_before = x.clone()
    alpha_before = alpha.clone()
    total_before = total.clone()
    ref_x = utils.to_reference(x, True)
    ref_alpha = utils.to_reference(alpha, True)
    ref_total = utils.to_reference(total, True)

    reference = torch.ops.aten._dirichlet_grad(ref_x, ref_alpha, ref_total)
    result = flag_gems._dirichlet_grad(x, alpha, total)

    assert result.shape == x.shape
    assert result.dtype == dtype
    utils.gems_assert_close(result, reference, dtype, atol=2e-3)
    utils.gems_assert_equal(x, x_before)
    utils.gems_assert_equal(alpha, alpha_before)
    utils.gems_assert_equal(total, total_before)


@pytest.mark.dirichlet_grad
def test_dirichlet_grad_rejects_nonbroadcastable_shape():
    x = torch.full((8, 4), 0.25, device=flag_gems.device)
    alpha = torch.ones((8, 4), device=flag_gems.device)
    total = torch.ones((3,), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems._dirichlet_grad(x, alpha, total)
