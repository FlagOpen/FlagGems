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

EMPTY_PERMUTED_LAYOUTS = [
    ((8,), [0]),
    ((4, 8), [0, 1]),
    ((4, 8), [1, 0]),
    ((2, 3, 4), [0, 1, 2]),
    ((2, 3, 4), [2, 0, 1]),
    ((2, 3, 4, 5), [0, 2, 3, 1]),
]


def _requested_device():
    requested = torch.device(flag_gems.device)
    if requested.index is None and requested.type != "cpu":
        requested = torch.device(
            requested.type, flag_gems.runtime.torch_device_fn.current_device()
        )
    return requested


def _assert_empty_metadata(result, reference, expected_device):
    # empty() intentionally returns uninitialized storage. Values are undefined,
    # so only the allocation contract may be compared. The candidate device is
    # checked against the requested target, not a possibly CPU-moved reference.
    assert result.shape == reference.shape
    assert result.stride() == reference.stride()
    assert result.dtype == reference.dtype
    assert result.layout == reference.layout
    assert result.device == expected_device
    assert result.numel() == reference.numel()
    assert result.requires_grad == reference.requires_grad


@pytest.mark.empty_permuted
@pytest.mark.parametrize("shape,physical_layout", EMPTY_PERMUTED_LAYOUTS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_empty_permuted(shape, physical_layout, dtype, caplog):
    # Values are uninitialized; compare only the shape/stride/layout contract.
    reference = torch.ops.aten.empty_permuted(
        shape, physical_layout, dtype=dtype, device=flag_gems.device
    )
    with caplog.at_level("DEBUG", logger="flag_gems.ops.empty_permuted"):
        result = flag_gems.empty_permuted(
            shape, physical_layout, dtype=dtype, device=flag_gems.device
        )

    assert (
        f"{utils.gems_log_prefix(flag_gems.empty_permuted)} EMPTY_PERMUTED"
        in caplog.text
    )
    _assert_empty_metadata(result, reference, _requested_device())
    assert torch.ops.aten.is_non_overlapping_and_dense(result)


@pytest.mark.empty
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_empty(shape, dtype):
    expected_device = "cpu" if cfg.TO_CPU else flag_gems.device
    reference = torch.empty(shape, dtype=dtype, device=expected_device)

    with flag_gems.use_gems():
        result = torch.empty(shape, dtype=dtype, device=flag_gems.device)

    _assert_empty_metadata(result, reference, _requested_device())


@pytest.mark.empty
def test_empty_default_dtype():
    expected_device = "cpu" if cfg.TO_CPU else flag_gems.device
    reference = torch.empty((10, 20), device=expected_device)

    with flag_gems.use_gems():
        result = torch.empty((10, 20), device=flag_gems.device)

    _assert_empty_metadata(result, reference, _requested_device())
    assert result.dtype == torch.get_default_dtype()


@pytest.mark.empty
@pytest.mark.parametrize("shape", [(0,), (2, 0, 3), (2, 3, 4, 5)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.bool, torch.complex64])
def test_empty_dtype_and_zero_size(shape, dtype):
    expected_device = "cpu" if cfg.TO_CPU else flag_gems.device
    reference = torch.empty(shape, dtype=dtype, device=expected_device)

    with flag_gems.use_gems():
        result = torch.empty(shape, dtype=dtype, device=flag_gems.device)

    _assert_empty_metadata(result, reference, _requested_device())
