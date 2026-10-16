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


def _reference(inp):
    return torch.special.erfinv(utils.to_reference(inp, True))


def _assert_inplace_result(
    result, inp, pointer, stride, storage_offset, reference, dtype
):
    assert result is inp
    assert result.data_ptr() == pointer
    assert result.stride() == stride
    assert result.storage_offset() == storage_offset
    assert result.shape == reference.shape
    assert result.dtype == dtype
    utils.gems_assert_close(result, reference, dtype, equal_nan=True)


@pytest.mark.special_erfinv_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_erfinv_valid_domain_(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
        -0.9, 0.9
    )
    pointer = inp.data_ptr()
    stride = inp.stride()
    storage_offset = inp.storage_offset()
    reference = _reference(inp.clone())

    result = flag_gems.special_erfinv_(inp)

    _assert_inplace_result(
        result, inp, pointer, stride, storage_offset, reference, dtype
    )


@pytest.mark.special_erfinv_
def test_special_erfinv_boundary_and_special_values_():
    inp = torch.tensor(
        [
            -1.25,
            -1.0,
            -0.0,
            0.0,
            1.0,
            1.25,
            float("inf"),
            float("-inf"),
            float("nan"),
        ],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    pointer = inp.data_ptr()
    stride = inp.stride()
    storage_offset = inp.storage_offset()
    reference = _reference(inp.clone())

    result = flag_gems.special_erfinv_(inp)

    _assert_inplace_result(
        result,
        inp,
        pointer,
        stride,
        storage_offset,
        reference,
        torch.float32,
    )


@pytest.mark.special_erfinv_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_erfinv_noncontiguous_view_(dtype):
    base = torch.empty(
        (3, 5, 8), dtype=dtype, device=flag_gems.device
    ).uniform_(-0.9, 0.9)
    inp = base[:, :, 1:7:2]
    assert not inp.is_contiguous()
    base_before = base.clone()
    outside_view = torch.ones_like(base, dtype=torch.bool)
    outside_view[:, :, 1:7:2] = False
    pointer = inp.data_ptr()
    stride = inp.stride()
    storage_offset = inp.storage_offset()
    reference = _reference(inp.clone())

    result = flag_gems.special_erfinv_(inp)

    _assert_inplace_result(
        result, inp, pointer, stride, storage_offset, reference, dtype
    )
    utils.gems_assert_equal(base[outside_view], base_before[outside_view])


@pytest.mark.special_erfinv_
def test_special_erfinv_rejects_leaf_requiring_grad_():
    inp = torch.tensor(
        [-0.75, -0.25, 0.25, 0.75],
        dtype=torch.float32,
        device=flag_gems.device,
        requires_grad=True,
    )
    before = inp.detach().clone()
    pointer = inp.data_ptr()

    with pytest.raises(RuntimeError):
        flag_gems.special_erfinv_(inp)

    assert inp.data_ptr() == pointer
    utils.gems_assert_equal(inp.detach(), before)
