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

import logging

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.adaptive_max_pool1d
@pytest.mark.parametrize("shape", [(2, 3, 16), (4, 8, 32), (1, 16, 64)])
@pytest.mark.parametrize("output_size", [8, 4, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_adaptive_max_pool1d(shape, output_size, dtype, caplog):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    with caplog.at_level(logging.DEBUG):
        expected, expected_indices = torch.nn.functional.adaptive_max_pool1d(
            ref_inp, output_size, return_indices=True
        )
        actual, actual_indices = flag_gems.adaptive_max_pool1d(inp, output_size)

    assert (
        f"{utils.gems_log_prefix(flag_gems.adaptive_max_pool1d)} ADAPTIVE_MAX_POOL1D"
        in caplog.text
    )
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual_indices.shape == expected_indices.shape
    assert actual_indices.dtype == torch.int64
    utils.gems_assert_close(actual, expected, dtype)
    utils.gems_assert_equal(actual_indices, expected_indices)


@pytest.mark.adaptive_max_pool1d
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adaptive_max_pool1d_list_output_size(dtype):
    torch.manual_seed(0)
    inp = torch.randn((2, 4, 20), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    expected, expected_indices = torch.nn.functional.adaptive_max_pool1d(
        ref_inp, [10], return_indices=True
    )
    actual, actual_indices = flag_gems.adaptive_max_pool1d(inp, [10])

    utils.gems_assert_close(actual, expected, dtype)
    utils.gems_assert_equal(actual_indices, expected_indices)


@pytest.mark.adaptive_max_pool1d
def test_adaptive_max_pool1d_global_pooling():
    torch.manual_seed(0)
    inp = torch.randn((1, 3, 100), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    expected, expected_indices = torch.nn.functional.adaptive_max_pool1d(
        ref_inp, 1, return_indices=True
    )
    actual, actual_indices = flag_gems.adaptive_max_pool1d(inp, 1)

    utils.gems_assert_close(actual, expected, torch.float32)
    utils.gems_assert_equal(actual_indices, expected_indices)

@pytest.mark.adaptive_max_pool1d
def test_adaptive_max_pool1d_explicit_autograd_rejection():
    torch.manual_seed(0)
    inp = torch.randn(
        (2, 3, 16), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    actual, _ = flag_gems.adaptive_max_pool1d(inp, 4)

    assert not actual.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(actual.sum(), inp)
