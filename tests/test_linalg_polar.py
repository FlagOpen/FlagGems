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

POLAR_SHAPES = [(8, 4)] if cfg.QUICK_MODE else [
    (3, 2),
    (8, 4),
    (16, 8),
    (2, 8, 4),
    (8, 16, 8),
]
ATOL = 1e-2


def _reference_linalg_polar(inp):
    svd_u, singular_values, vh = torch.linalg.svd(inp, full_matrices=False)
    polar_u = svd_u @ vh
    polar_h = vh.mH @ (singular_values.unsqueeze(-1) * vh)
    return polar_u, 0.5 * (polar_h + polar_h.mH)


TORCH_LINALG_POLAR = (
    torch.ops.aten.linalg_polar.default
    if hasattr(torch.ops.aten, "linalg_polar")
    else _reference_linalg_polar
)


def _assert_polar_properties(inp, polar_u, polar_h):
    utils.gems_assert_close(polar_u @ polar_h, inp, torch.float32, atol=ATOL)
    utils.gems_assert_close(
        polar_h, utils.to_reference(polar_h.mT, True), torch.float32, atol=ATOL
    )
    gram = polar_u.mT @ polar_u
    eye = torch.eye(polar_u.shape[-1], dtype=polar_u.dtype, device=flag_gems.device)
    utils.gems_assert_close(
        gram,
        utils.to_reference(eye.expand_as(gram), True),
        torch.float32,
        atol=ATOL,
    )


@pytest.mark.linalg_polar
@pytest.mark.parametrize("shape", POLAR_SHAPES)
def test_linalg_polar(shape):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    inp_before = inp.clone()
    ref_inp = utils.to_reference(inp, True)
    _, ref_h = TORCH_LINALG_POLAR(ref_inp)

    polar_u, polar_h = flag_gems.linalg_polar(inp)

    assert polar_u.shape == inp.shape
    assert polar_h.shape == (*inp.shape[:-2], inp.shape[-1], inp.shape[-1])
    assert polar_u.is_contiguous()
    assert polar_h.is_contiguous()
    assert polar_u is not inp
    utils.gems_assert_equal(inp, inp_before)
    utils.gems_assert_close(polar_h, ref_h, torch.float32, atol=ATOL)
    _assert_polar_properties(ref_inp, polar_u, polar_h)


@pytest.mark.linalg_polar
def test_linalg_polar_noncontiguous():
    torch.manual_seed(0)
    inp = torch.randn((4, 16), device=flag_gems.device).mT
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp, True)

    polar_u, polar_h = flag_gems.linalg_polar(inp)

    _assert_polar_properties(ref_inp, polar_u, polar_h)


@pytest.mark.linalg_polar
@pytest.mark.parametrize("shape", [(0, 0), (3, 0), (0, 5, 3)])
def test_linalg_polar_empty(shape):
    inp = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_u, ref_h = TORCH_LINALG_POLAR(ref_inp)

    polar_u, polar_h = flag_gems.linalg_polar(inp)

    assert polar_u.shape == ref_u.shape
    assert polar_h.shape == ref_h.shape
    assert polar_u.dtype == inp.dtype
    assert polar_h.dtype == inp.dtype
    assert polar_u.is_contiguous()
    assert polar_h.is_contiguous()


@pytest.mark.linalg_polar
def test_linalg_polar_rank_deficient():
    inp = torch.zeros((8, 4), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    polar_u, polar_h = flag_gems.linalg_polar(inp)

    utils.gems_assert_equal(polar_h, torch.zeros_like(ref_inp[:4]))
    utils.gems_assert_equal(polar_u @ polar_h, ref_inp)


@pytest.mark.linalg_polar
@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((3,), torch.float32),
        ((3, 5), torch.float32),
        ((5, 3), torch.float64),
    ],
)
def test_linalg_polar_invalid_input(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.linalg_polar(inp)


@pytest.mark.linalg_polar
def test_linalg_polar_autograd_is_explicitly_unsupported():
    torch.manual_seed(0)
    inp = torch.randn((8, 4), dtype=torch.float32, device=flag_gems.device)
    inp.requires_grad_(True)

    polar_u, polar_h = flag_gems.linalg_polar(inp)

    assert not polar_u.requires_grad
    assert not polar_h.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(polar_h.square().sum(), inp)
