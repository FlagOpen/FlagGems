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


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


def _nan_input(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp.reshape(-1)[::5] = float("nan")
    return inp


@pytest.mark.nanmean_out
@pytest.mark.parametrize("dim,keepdim", [(0, False), (1, True), ((0, 1), False)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nanmean_out_content_identity(dim, keepdim, dtype):
    inp = _nan_input((7, 11), dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_shape = torch.nanmean(ref_inp, dim=dim, keepdim=keepdim).shape
    ref_out = torch.empty(ref_shape, dtype=dtype, device=ref_inp.device)
    reference = torch.nanmean(ref_inp, dim=dim, keepdim=keepdim, out=ref_out)
    out = torch.empty(ref_shape, dtype=dtype, device=inp.device)
    pointer = out.data_ptr()

    result = flag_gems.nanmean_out(inp, dim=dim, keepdim=keepdim, out=out)

    assert result is out
    assert result.data_ptr() == pointer
    assert reference is ref_out
    reduction_size = inp.numel() if isinstance(dim, tuple) else inp.shape[dim]
    utils.gems_assert_close(
        result, reference, dtype, equal_nan=True, reduce_dim=reduction_size
    )


@pytest.mark.nanmean_out
def test_nanmean_out_resizes_and_accepts_output_cast():
    inp = _nan_input((4, 8), torch.float32)
    ref_inp = utils.to_reference(inp)
    out = torch.empty((0,), dtype=torch.float16, device=inp.device)
    ref_out = torch.empty((0,), dtype=torch.float16, device=ref_inp.device)
    reference = torch.nanmean(ref_inp, dim=1, out=ref_out)

    result = flag_gems.nanmean_out(inp, dim=1, out=out)

    assert result is out
    assert result.shape == (4,)
    assert result.dtype == torch.float16
    utils.gems_assert_close(
        result, reference, torch.float16, equal_nan=True, reduce_dim=8
    )


@pytest.mark.nanmean_out
def test_nanmean_out_noncontiguous_buffer():
    inp = _nan_input((65, 129), torch.float32)
    ref_inp = utils.to_reference(inp)
    out = torch.empty((65, 2), dtype=torch.float16, device=inp.device)[:, 0]
    ref_out = torch.empty((65, 2), dtype=torch.float16, device=ref_inp.device)[:, 0]
    reference = torch.nanmean(ref_inp, dim=1, out=ref_out)

    result = flag_gems.nanmean_out(inp, dim=1, out=out)

    assert result is out
    assert result.stride() == ref_out.stride()
    utils.gems_assert_close(
        result, reference, torch.float16, equal_nan=True, reduce_dim=129
    )


@pytest.mark.nanmean_out
def test_nanmean_out_rejects_dtype_mismatch_and_autograd():
    inp = torch.randn((4, 8), device=flag_gems.device)
    out = torch.empty((4,), dtype=torch.float16, device=inp.device)
    with pytest.raises(RuntimeError):
        flag_gems.nanmean_out(inp, dim=1, dtype=torch.float32, out=out)

    grad_input = inp.requires_grad_(True)
    with pytest.raises(RuntimeError):
        flag_gems.nanmean_out(
            grad_input, dim=1, out=torch.empty(4, device=inp.device)
        )
