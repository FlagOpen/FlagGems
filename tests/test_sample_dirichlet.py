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


@pytest.mark.sample_dirichlet
@pytest.mark.parametrize("shape", [(3,), (10, 5), (20, 10), (4, 4096), (64, 1)])
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_sample_dirichlet_simplex_contract(shape, dtype):
    torch.manual_seed(42)
    alpha = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 5.0 + 0.5
    alpha_before = alpha.clone()

    result = flag_gems._sample_dirichlet(alpha)

    assert result.shape == alpha.shape
    assert result.dtype == dtype
    assert torch.all(result >= 0)
    sums = result.sum(dim=-1)
    expected = utils.to_reference(torch.ones_like(sums), True)
    utils.gems_assert_close(sums, expected, dtype)
    utils.gems_assert_equal(alpha, alpha_before)


@pytest.mark.sample_dirichlet
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_sample_dirichlet_uniform_mean(dtype):
    alpha = torch.ones((1000, 5), dtype=dtype, device=flag_gems.device)
    generator = torch.Generator(device=flag_gems.device).manual_seed(42)

    result = flag_gems._sample_dirichlet(alpha, generator=generator)

    means = result.float().mean(dim=0)
    expected = utils.to_reference(torch.full_like(means, 0.2), True)
    utils.gems_assert_close(means, expected, torch.float32, atol=0.05)


@pytest.mark.sample_dirichlet
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_sample_dirichlet_skewed_alpha(dtype):
    alpha = torch.tensor(
        [[10.0, 1.0, 1.0]], dtype=dtype, device=flag_gems.device
    ).expand(500, 3)
    generator = torch.Generator(device=flag_gems.device).manual_seed(42)

    result = flag_gems._sample_dirichlet(alpha, generator=generator)
    means = result.float().mean(dim=0)

    assert means[0] > means[1]
    assert means[0] > means[2]


@pytest.mark.sample_dirichlet
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_sample_dirichlet_tiny_alpha(dtype):
    alpha = torch.full((256, 8), 1e-3, dtype=dtype, device=flag_gems.device)

    result = flag_gems._sample_dirichlet(alpha)

    assert torch.all(result >= 0)
    sums = result.sum(dim=-1)
    utils.gems_assert_close(
        sums, utils.to_reference(torch.ones_like(sums), True), dtype
    )


@pytest.mark.sample_dirichlet
def test_sample_dirichlet_generator_contract():
    alpha = torch.full((256, 8), 2.0, device=flag_gems.device)
    generator = torch.Generator(device=flag_gems.device)

    generator.manual_seed(1234)
    first = flag_gems._sample_dirichlet(alpha, generator=generator)
    second = flag_gems._sample_dirichlet(alpha, generator=generator)
    assert not torch.equal(first, second)

    generator.manual_seed(1234)
    replay = flag_gems._sample_dirichlet(alpha, generator=generator)
    utils.gems_assert_equal(first, replay)


@pytest.mark.sample_dirichlet
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
def test_sample_dirichlet_rejects_non_floating(dtype):
    alpha = torch.ones((8, 4), dtype=dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems._sample_dirichlet(alpha)
