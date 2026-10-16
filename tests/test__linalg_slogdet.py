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

SLOGDET_SHAPES = [(2, 3, 3), (4, 4), (8, 8), (16, 16), (32, 32)]


@pytest.mark.underscore_linalg_slogdet
@pytest.mark.linalg_slogdet
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test__linalg_slogdet(shape, dtype):
    n = shape[-1]
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = inp + torch.eye(n, dtype=dtype, device=flag_gems.device) * n
    ref_inp = utils.to_reference(inp, True)

    ref_sign, ref_logabsdet = torch.linalg.slogdet(ref_inp)
    sign, logabsdet, lu, pivots = torch.ops.aten._linalg_slogdet(inp)

    utils.gems_assert_close(sign, ref_sign, dtype)
    utils.gems_assert_close(logabsdet, ref_logabsdet, dtype, reduce_dim=n)
    assert lu.shape == inp.shape
    assert pivots.shape == inp.shape[:-1]
    assert pivots.dtype in (torch.int32, torch.int64)


@pytest.mark.underscore_linalg_slogdet
@pytest.mark.linalg_slogdet
def test__linalg_slogdet_rejects_non_square():
    torch.manual_seed(0)
    inp = torch.randn((3, 4), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten._linalg_slogdet(inp)


@pytest.mark.underscore_linalg_slogdet
@pytest.mark.linalg_slogdet
def test__linalg_slogdet_autograd_is_explicitly_unsupported():
    torch.manual_seed(0)
    inp = torch.randn((4, 4), dtype=torch.float32, device=flag_gems.device)
    inp = (inp + torch.eye(4, device=inp.device) * 4).requires_grad_(True)

    sign, logabsdet, _, _ = flag_gems._linalg_slogdet(inp)

    assert not sign.requires_grad
    assert not logabsdet.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(logabsdet.sum(), inp)
