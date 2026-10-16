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

def _inputs(dtype):
    torch.manual_seed(0)
    inp = torch.randn((2, 16, 9, 9), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((8, 16, 3, 3), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((8,), dtype=dtype, device=flag_gems.device)
    return inp, weight, bias


def _reference(inp, weight, bias):
    return torch.ops.aten.convolution(
        utils.to_reference(inp, True), utils.to_reference(weight, True),
        utils.to_reference(bias, True), [1, 1], [1, 1], [1, 1],
        False, [0, 0], 1
    ).to(inp.dtype)


@pytest.mark.convolution_overrideable_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_convolution_overrideable_out_identity_and_storage(dtype):
    inp, weight, bias = _inputs(dtype)
    ref = _reference(inp, weight, bias)
    backing = torch.empty(
        (ref.shape[0], ref.shape[1], ref.shape[2], ref.shape[3] * 2),
        dtype=dtype, device=flag_gems.device
    )
    out = backing[..., ::2]
    pointer = out.data_ptr()
    stride = out.stride()
    offset = out.storage_offset()
    result = flag_gems.convolution_overrideable_out(
        inp, weight, bias, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, out=out
    )
    assert result is out
    assert out.data_ptr() == pointer
    assert out.stride() == stride
    assert out.storage_offset() == offset
    utils.gems_assert_close(out, ref, dtype, reduce_dim=144)


@pytest.mark.convolution_overrideable_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_convolution_overrideable_out_resizes_empty(dtype):
    inp, weight, bias = _inputs(dtype)
    ref = _reference(inp, weight, bias)
    out = torch.empty((0,), dtype=dtype, device=flag_gems.device)
    result = flag_gems.convolution_overrideable_out(
        inp, weight, bias, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, out=out
    )
    assert result is out
    assert out.shape == ref.shape
    assert out.dtype == dtype
    utils.gems_assert_close(out, ref, dtype, reduce_dim=144)


@pytest.mark.convolution_overrideable_out
def test_convolution_overrideable_out_rejects_dtype_mismatch():
    inp, weight, bias = _inputs(torch.float32)
    out = torch.empty((2, 8, 9, 9), dtype=torch.int32, device=flag_gems.device)
    snapshot = out.clone()
    pointer, stride = out.data_ptr(), out.stride()
    with pytest.raises(RuntimeError):
        flag_gems.convolution_overrideable_out(
            inp, weight, bias, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, out=out
        )
    assert out.data_ptr() == pointer
    assert out.stride() == stride
    utils.gems_assert_equal(out, snapshot)


@pytest.mark.convolution_overrideable_out
def test_convolution_overrideable_out_rejects_device_mismatch():
    if str(flag_gems.device) == "cpu":
        pytest.skip("device mismatch requires a non-CPU target device")
    inp, weight, bias = _inputs(torch.float32)
    out = torch.empty((2, 8, 9, 9), dtype=torch.float32, device="cpu")
    with pytest.raises(RuntimeError):
        flag_gems.convolution_overrideable_out(
            inp, weight, bias, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, out=out
        )
