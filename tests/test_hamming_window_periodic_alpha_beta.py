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


_WINDOW_LENGTHS = [0, 1, 2, 16, 256, 512]
_COEFFICIENTS = [
    pytest.param(0.54, 0.46, id="standard"),
    pytest.param(0.5, 0.5, id="hann-like"),
]


def _device():
    return "cpu" if cfg.TO_CPU else flag_gems.device


def _requested_device():
    requested = torch.device(_device())
    if requested.index is None and requested.type != "cpu":
        requested = torch.device(
            requested.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    return requested


def _assert_window(result, reference, window_length, dtype):
    assert result.shape == (window_length,)
    assert result.dtype == dtype
    assert result.layout == torch.strided
    assert result.device == _requested_device()
    utils.gems_assert_close(result, reference, dtype=dtype)


@pytest.mark.hamming_window_periodic_alpha_beta
@pytest.mark.parametrize("window_length", _WINDOW_LENGTHS)
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("alpha,beta", _COEFFICIENTS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hamming_window_periodic_alpha_beta(
    window_length, periodic, alpha, beta, dtype
):
    reference = torch.hamming_window(
        window_length,
        periodic=periodic,
        alpha=alpha,
        beta=beta,
        dtype=dtype,
        layout=torch.strided,
        device=_device(),
        pin_memory=False,
    )
    reference = utils.to_reference(reference, True)
    result = flag_gems.hamming_window_periodic_alpha_beta(
        window_length,
        periodic,
        alpha,
        beta,
        dtype=dtype,
        layout=torch.strided,
        device=_device(),
        pin_memory=False,
    )

    _assert_window(result, reference, window_length, dtype)


@pytest.mark.hamming_window_periodic_alpha_beta
@pytest.mark.parametrize(
    "alpha,beta",
    [
        pytest.param(0.0, 0.0, id="zeros"),
        pytest.param(1.0, 0.0, id="constant-one"),
        pytest.param(-0.25, 0.75, id="signed-alpha"),
    ],
)
def test_hamming_window_periodic_alpha_beta_coefficient_boundaries(alpha, beta):
    reference = torch.hamming_window(
        17,
        periodic=False,
        alpha=alpha,
        beta=beta,
        dtype=torch.float32,
        device=_device(),
    )
    reference = utils.to_reference(reference, True)
    result = flag_gems.hamming_window_periodic_alpha_beta(
        17,
        False,
        alpha,
        beta,
        dtype=torch.float32,
        device=_device(),
    )

    _assert_window(result, reference, 17, torch.float32)


@pytest.mark.hamming_window_periodic_alpha_beta
def test_hamming_window_periodic_alpha_beta_default_dtype():
    reference = torch.hamming_window(
        33,
        periodic=True,
        alpha=0.54,
        beta=0.46,
        device=_device(),
    )
    reference = utils.to_reference(reference, True)
    result = flag_gems.hamming_window_periodic_alpha_beta(
        33,
        True,
        0.54,
        0.46,
        device=_device(),
    )

    _assert_window(result, reference, 33, torch.get_default_dtype())


@pytest.mark.hamming_window_periodic_alpha_beta
def test_hamming_window_periodic_alpha_beta_requires_grad_dispatch():
    reference = torch.hamming_window(
        33,
        periodic=False,
        alpha=0.54,
        beta=0.46,
        dtype=torch.float32,
        device=_device(),
        requires_grad=True,
    )
    reference = utils.to_reference(reference, True)
    with flag_gems.use_gems():
        result = torch.hamming_window(
            33,
            periodic=False,
            alpha=0.54,
            beta=0.46,
            dtype=torch.float32,
            device=_device(),
            requires_grad=True,
        )

    assert result.requires_grad
    _assert_window(result, reference, 33, torch.float32)


@pytest.mark.hamming_window_periodic_alpha_beta
def test_hamming_window_periodic_alpha_beta_rejects_negative_length():
    with pytest.raises(AssertionError, match="window_length must be non-negative"):
        flag_gems.hamming_window_periodic_alpha_beta(
            -1, True, 0.54, 0.46, dtype=torch.float32, device=_device()
        )


@pytest.mark.hamming_window_periodic_alpha_beta
def test_hamming_window_periodic_alpha_beta_rejects_integer_dtype():
    with pytest.raises(AssertionError, match="only supports floating point dtype"):
        flag_gems.hamming_window_periodic_alpha_beta(
            16, True, 0.54, 0.46, dtype=torch.int32, device=_device()
        )


@pytest.mark.hamming_window_periodic_alpha_beta
def test_hamming_window_periodic_alpha_beta_rejects_non_strided_layout():
    with pytest.raises(ValueError, match="Only strided layout is supported"):
        flag_gems.hamming_window_periodic_alpha_beta(
            16,
            True,
            0.54,
            0.46,
            dtype=torch.float32,
            layout=torch.sparse_coo,
            device=_device(),
        )
