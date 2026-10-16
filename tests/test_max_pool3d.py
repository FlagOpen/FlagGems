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


MAXPOOL3D_CONFIGS = [
    ((4, 3, 16, 16, 16), 3, 2, 1, 1, False),
    ((8, 16, 12, 14, 14), (2, 3, 3), (1, 2, 2), (0, 1, 1), 1, False),
    ((2, 4, 15, 15, 15), 3, 2, 1, 1, True),
    ((1, 1, 9, 9, 9), 2, 1, 0, 2, False),
    ((1, 64, 8, 28, 28), 3, 2, 1, 1, False),
    ((2, 8, 8, 16, 16), 2, 2, 0, 1, False),
    ((2, 8, 10, 16, 20), 2, 2, (0, 1, 0), 1, False),
    ((1, 1, 5, 5, 5), 2, 1, 0, 1, False),
    ((8, 16, 8, 8, 8), 3, 1, 1, 1, False),
]


@pytest.mark.max_pool3d
@pytest.mark.parametrize(
    "shape,kernel_size,stride,padding,dilation,ceil_mode", MAXPOOL3D_CONFIGS
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_max_pool3d(shape, kernel_size, stride, padding, dilation, ceil_mode, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    expected = torch.max_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    actual = flag_gems.max_pool3d(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    utils.gems_assert_close(actual, expected, dtype)


@pytest.mark.max_pool3d
@pytest.mark.parametrize("shape", [(2, 3, 7, 9, 11), (3, 7, 9, 11)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_max_pool3d_default_stride(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    expected = torch.max_pool3d(ref_inp, kernel_size=2)
    actual = flag_gems.max_pool3d(inp, kernel_size=2)

    utils.gems_assert_close(actual, expected, dtype)

@pytest.mark.max_pool3d
def test_max_pool3d_explicit_autograd_rejection():
    torch.manual_seed(0)
    inp = torch.randn(
        (1, 2, 6, 6, 6),
        dtype=torch.float32,
        device=flag_gems.device,
        requires_grad=True,
    )
    actual = flag_gems.max_pool3d(inp, kernel_size=2, stride=2)

    assert not actual.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(actual.sum(), inp)
