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

DIMENSIONS = [1, 2, 5, 10]
N_VALUES = [0, 1, 10, 100]
NUM_GENERATED_VALUES = [0, 10, 100]
MAXBIT = 30


def _make_inputs(dimension, seed=0):
    torch.manual_seed(seed)
    quasi = torch.randint(
        0, 2**30, (dimension,), dtype=torch.int64, device=flag_gems.device
    )
    sobolstate = torch.randint(
        0,
        2**30,
        (dimension, MAXBIT),
        dtype=torch.int64,
        device=flag_gems.device,
    )
    return quasi, sobolstate


@pytest.mark.sobol_engine_ff_
@pytest.mark.parametrize("dimension", DIMENSIONS)
@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("num_generated", NUM_GENERATED_VALUES)
def test_sobol_engine_ff_mutation_and_identity(dimension, n, num_generated):
    quasi, sobolstate = _make_inputs(dimension)
    quasi_before = quasi.clone()
    sobolstate_before = sobolstate.clone()
    quasi_pointer = quasi.data_ptr()

    ref_quasi = quasi_before.cpu()
    ref_sobolstate = sobolstate_before.cpu()
    ref_result = torch._sobol_engine_ff_(
        ref_quasi, n, ref_sobolstate, dimension, num_generated
    )
    result = flag_gems.ops._sobol_engine_ff_(
        quasi, n, sobolstate, dimension, num_generated
    )

    assert result is quasi
    assert result.data_ptr() == quasi_pointer
    assert ref_result is ref_quasi
    utils.gems_assert_equal(result.cpu(), ref_result)
    utils.gems_assert_equal(sobolstate, sobolstate_before)
    assert result.shape == (dimension,)
    assert result.dtype == torch.int64


@pytest.mark.sobol_engine_ff_
def test_sobol_engine_ff_seed_reproducibility():
    quasi_a, state_a = _make_inputs(8, seed=2026)
    quasi_b, state_b = _make_inputs(8, seed=2026)

    result_a = flag_gems.ops._sobol_engine_ff_(quasi_a, 31, state_a, 8, 17)
    result_b = flag_gems.ops._sobol_engine_ff_(quasi_b, 31, state_b, 8, 17)

    utils.gems_assert_equal(result_a, result_b)
    utils.gems_assert_equal(state_a, state_b)
