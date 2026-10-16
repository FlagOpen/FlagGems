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

DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES

CHAIN_CASES = [
    ((64, 32, 128), False, False),
    ((64, 32, 128, 16), False, False),
    ((64, 128, 32, 16), False, False),
    ((1, 64, 32, 128, 16), True, False),
    ((64, 32, 128, 64, 32, 1), False, True),
    ((1, 64, 1), True, True),
]

def _make_chain(dimensions, first_vector, last_vector, dtype):
    tensors = [
        torch.randn(
            dimensions[index],
            dimensions[index + 1],
            dtype=dtype,
            device=flag_gems.device,
        )
        for index in range(len(dimensions) - 1)
    ]
    if first_vector:
        tensors[0] = tensors[0].squeeze(0)
    if last_vector:
        tensors[-1] = tensors[-1].squeeze(1)
    return tensors

def _reference_multi_dot(tensors, *, out=None):
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        return torch.linalg.multi_dot(tensors, out=out)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32

@pytest.mark.linalg_multi_dot_out
@pytest.mark.parametrize("dimensions,first_vector,last_vector", CHAIN_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_accuracy_linalg_multi_dot_out(dimensions, first_vector, last_vector, dtype):
    torch.manual_seed(0)
    tensors = _make_chain(dimensions, first_vector, last_vector, dtype)
    reference_tensors = [utils.to_reference(tensor, True) for tensor in tensors]
    reference_out = torch.empty(
        0, dtype=reference_tensors[0].dtype, device=reference_tensors[0].device
    )
    reference = _reference_multi_dot(reference_tensors, out=reference_out)

    out = torch.empty(0, dtype=dtype, device=flag_gems.device)
    result = flag_gems.linalg_multi_dot_out(tensors, out=out)

    assert reference is reference_out
    assert result is out
    assert tuple(out.shape) == tuple(reference.shape)
    utils.gems_assert_close(out, reference, dtype, reduce_dim=max(dimensions))

@pytest.mark.linalg_multi_dot_out
@pytest.mark.parametrize("dtype", DTYPES)
def test_accuracy_linalg_multi_dot_out_noncontiguous(dtype):
    torch.manual_seed(0)
    dimensions = (64, 32, 128)
    tensors = _make_chain(dimensions, False, False, dtype)
    reference_tensors = [utils.to_reference(tensor, True) for tensor in tensors]
    reference_out = torch.empty(
        dimensions[-1],
        dimensions[0],
        dtype=reference_tensors[0].dtype,
        device=reference_tensors[0].device,
    ).t()
    reference = _reference_multi_dot(reference_tensors, out=reference_out)
    out = torch.empty(
        dimensions[-1], dimensions[0], dtype=dtype, device=flag_gems.device
    ).t()
    out_pointer = out.data_ptr()

    result = flag_gems.linalg_multi_dot_out(tensors, out=out)

    assert reference is reference_out
    assert result is out
    assert out.data_ptr() == out_pointer
    assert not out.is_contiguous()
    utils.gems_assert_close(out, reference, dtype, reduce_dim=max(dimensions))


@pytest.mark.linalg_multi_dot_out
def test_linalg_multi_dot_out_dtype_conversion():
    torch.manual_seed(0)
    dimensions = (8, 4, 6)
    tensors = _make_chain(dimensions, False, False, torch.float32)
    reference_tensors = [
        utils.to_reference(tensor, True).to(torch.float32) for tensor in tensors
    ]
    reference_out = torch.empty(
        (dimensions[0], dimensions[-1]),
        dtype=torch.float64,
        device=reference_tensors[0].device,
    )
    reference = _reference_multi_dot(reference_tensors, out=reference_out)
    out = torch.empty(
        (dimensions[0], dimensions[-1]),
        dtype=torch.float64,
        device=flag_gems.device,
    )
    out_pointer = out.data_ptr()

    result = flag_gems.linalg_multi_dot_out(tensors, out=out)

    assert reference is reference_out
    assert result is out
    assert out.data_ptr() == out_pointer
    assert out.dtype == torch.float64
    utils.gems_assert_close(out, reference, torch.float32, reduce_dim=max(dimensions))


@pytest.mark.linalg_multi_dot_out
def test_linalg_multi_dot_out_rejects_integer_out():
    torch.manual_seed(0)
    dimensions = (8, 4, 6)
    tensors = _make_chain(dimensions, False, False, torch.float32)
    reference_tensors = [
        utils.to_reference(tensor, True).to(torch.float32) for tensor in tensors
    ]
    reference_out = torch.empty(
        (dimensions[0], dimensions[-1]),
        dtype=torch.int64,
        device=reference_tensors[0].device,
    )
    with pytest.raises(RuntimeError):
        _reference_multi_dot(reference_tensors, out=reference_out)

    out = torch.empty(
        (dimensions[0], dimensions[-1]),
        dtype=torch.int64,
        device=flag_gems.device,
    )
    with pytest.raises(RuntimeError):
        flag_gems.linalg_multi_dot_out(tensors, out=out)
