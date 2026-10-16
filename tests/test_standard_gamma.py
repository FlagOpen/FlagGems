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


@pytest.mark.standard_gamma
@pytest.mark.parametrize("shape", [*utils.POINTWISE_SHAPES, (0,), (17, 1)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_standard_gamma_metadata_and_domain(shape, dtype):
    torch.manual_seed(0)
    alpha = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 5.0 + 0.5
    alpha_before = alpha.clone()

    result = flag_gems.standard_gamma(alpha)

    assert result.shape == alpha.shape
    assert result.dtype == dtype
    assert result.device == alpha.device
    assert torch.all(result >= 0)
    utils.gems_assert_equal(alpha, alpha_before)


@pytest.mark.standard_gamma
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_standard_gamma_distribution_moments(alpha):
    inp = torch.full((10000,), alpha, dtype=torch.float32, device=flag_gems.device)
    generator = torch.Generator(device=flag_gems.device).manual_seed(12345)

    result = flag_gems.standard_gamma(inp, generator=generator)

    mean = result.mean().item()
    variance = result.var().item()
    assert abs(mean - alpha) < alpha * 0.2 + 0.5
    assert abs(variance - alpha) < alpha * 0.3 + 1.0


@pytest.mark.standard_gamma
@pytest.mark.parametrize("alpha", [1e-4, 1e-3, 1e-2, 0.05])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_standard_gamma_tiny_alpha_is_positive(alpha, dtype):
    inp = torch.full((4096,), alpha, dtype=dtype, device=flag_gems.device)

    result = flag_gems.standard_gamma(inp)

    assert result.dtype == dtype
    assert torch.all(result > 0)


@pytest.mark.standard_gamma
def test_standard_gamma_generator_contract():
    inp = torch.full((2048,), 2.0, dtype=torch.float32, device=flag_gems.device)
    generator = torch.Generator(device=flag_gems.device)

    generator.manual_seed(777)
    first = flag_gems.standard_gamma(inp, generator=generator)
    second = flag_gems.standard_gamma(inp, generator=generator)
    assert not torch.equal(first, second)

    generator.manual_seed(777)
    replay = flag_gems.standard_gamma(inp, generator=generator)
    utils.gems_assert_equal(first, replay)


@pytest.mark.standard_gamma
@pytest.mark.parametrize(
    "dtype", [torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]
)
def test_standard_gamma_rejects_non_floating(dtype):
    inp = torch.ones((16,), dtype=dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.standard_gamma(inp)


@pytest.mark.standard_gamma
def test_standard_gamma_rejects_float64():
    inp = torch.rand((16,), dtype=torch.float64, device=flag_gems.device) + 0.5

    with pytest.raises(RuntimeError):
        flag_gems.standard_gamma(inp)
