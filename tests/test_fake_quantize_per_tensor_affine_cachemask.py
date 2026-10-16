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


def _seed():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def _reference(tensor):
    return utils.to_reference(tensor, True).to(tensor.dtype)


@pytest.mark.fake_quantize_per_tensor_affine_cachemask
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("scale", [0.03, 0.125])
@pytest.mark.parametrize("quant_min, quant_max", [(0, 255), (-128, 127)])
def test_fake_quantize_per_tensor_affine_cachemask(
    shape, dtype, scale, quant_min, quant_max
):
    _seed()
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 4
    snapshot = inp.clone()
    zero_point = 3 if quant_min == 0 else -7
    ref_output, ref_mask = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
        _reference(inp), scale, zero_point, quant_min, quant_max
    )

    output, mask = flag_gems.fake_quantize_per_tensor_affine_cachemask(
        inp, scale, zero_point, quant_min, quant_max
    )

    utils.gems_assert_equal(output, ref_output)
    utils.gems_assert_equal(mask, ref_mask)
    utils.gems_assert_equal(inp, snapshot)
    assert output.shape == inp.shape
    assert output.dtype == inp.dtype
    assert mask.shape == inp.shape
    assert mask.dtype == torch.bool


@pytest.mark.fake_quantize_per_tensor_affine_cachemask
@pytest.mark.parametrize("scale", [0.0, 1.0])
def test_fake_quantize_per_tensor_affine_cachemask_boundaries(scale):
    inp = torch.tensor(
        [float("-inf"), -3.5, -0.5, 0.5, 3.5, float("inf"), float("nan")],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_output, ref_mask = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
        _reference(inp), scale, 0, -2, 2
    )

    output, mask = flag_gems.fake_quantize_per_tensor_affine_cachemask(
        inp, scale, 0, -2, 2
    )

    utils.gems_assert_equal(output, ref_output)
    utils.gems_assert_equal(mask, ref_mask)


@pytest.mark.fake_quantize_per_tensor_affine_cachemask
@pytest.mark.parametrize("empty", [False, True])
def test_fake_quantize_per_tensor_affine_cachemask_layout_and_empty(empty):
    _seed()
    inp = (
        torch.empty((2, 0, 3), device=flag_gems.device)
        if empty
        else torch.randn((8, 16), device=flag_gems.device).T
    )
    snapshot = inp.clone()
    ref_output, ref_mask = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
        _reference(inp), 0.1, 0, 0, 255
    )

    output, mask = flag_gems.fake_quantize_per_tensor_affine_cachemask(
        inp, 0.1, 0, 0, 255
    )

    utils.gems_assert_equal(output, ref_output)
    utils.gems_assert_equal(mask, ref_mask)
    utils.gems_assert_equal(inp, snapshot)
    assert output.shape == inp.shape
    assert mask.shape == inp.shape


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_out
def test_fake_quantize_per_tensor_affine_cachemask_out_preserves_buffers():
    _seed()
    inp = torch.randn((8, 16), device=flag_gems.device)
    snapshot = inp.clone()
    ref_inp = _reference(inp)
    out0 = torch.empty((16, 8), dtype=inp.dtype, device=flag_gems.device).T
    out1 = torch.empty((16, 8), dtype=torch.bool, device=flag_gems.device).T
    ref_out0 = torch.empty((16, 8), dtype=ref_inp.dtype, device=ref_inp.device).T
    ref_out1 = torch.empty((16, 8), dtype=torch.bool, device=ref_inp.device).T
    pointer0, pointer1 = out0.data_ptr(), out1.data_ptr()
    stride0, stride1 = out0.stride(), out1.stride()
    offset0, offset1 = out0.storage_offset(), out1.storage_offset()

    ref_output, ref_mask = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask.out(
        ref_inp, 0.03, 3, 0, 255, out0=ref_out0, out1=ref_out1
    )
    result = flag_gems.fake_quantize_per_tensor_affine_cachemask_out(
        inp, 0.03, 3, 0, 255, out0=out0, out1=out1
    )

    assert result[0] is out0
    assert result[1] is out1
    assert out0.data_ptr() == pointer0
    assert out1.data_ptr() == pointer1
    assert out0.stride() == stride0
    assert out1.stride() == stride1
    assert out0.storage_offset() == offset0
    assert out1.storage_offset() == offset1
    utils.gems_assert_equal(out0, ref_output)
    utils.gems_assert_equal(out1, ref_mask)
    utils.gems_assert_equal(inp, snapshot)


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_out
def test_fake_quantize_per_tensor_affine_cachemask_out_resizes_empty_buffers():
    _seed()
    inp = torch.randn((4, 7), device=flag_gems.device)
    ref_output, ref_mask = torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
        _reference(inp), 0.125, 3, 0, 255
    )
    out0 = torch.empty(0, dtype=inp.dtype, device=flag_gems.device)
    out1 = torch.empty(0, dtype=torch.bool, device=flag_gems.device)

    result = flag_gems.fake_quantize_per_tensor_affine_cachemask_out(
        inp, 0.125, 3, 0, 255, out0=out0, out1=out1
    )

    assert result[0] is out0
    assert result[1] is out1
    assert out0.shape == inp.shape
    assert out1.shape == inp.shape
    utils.gems_assert_equal(out0, ref_output)
    utils.gems_assert_equal(out1, ref_mask)


@pytest.mark.fake_quantize_per_tensor_affine_cachemask
@pytest.mark.parametrize("zero_point, quant_min, quant_max", [(256, 0, 255), (0, 2, 1)])
def test_fake_quantize_per_tensor_affine_cachemask_invalid_qparams(
    zero_point, quant_min, quant_max
):
    inp = torch.ones(2, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.fake_quantize_per_tensor_affine_cachemask(
            inp, 0.1, zero_point, quant_min, quant_max
        )


@pytest.mark.fake_quantize_per_tensor_affine_cachemask_out
@pytest.mark.parametrize("bad_output", [0, 1])
def test_fake_quantize_per_tensor_affine_cachemask_out_rejects_dtype(bad_output):
    inp = torch.ones(2, device=flag_gems.device)
    out0 = torch.empty(2, dtype=torch.float64, device=flag_gems.device)
    out1 = torch.empty(2, dtype=torch.uint8, device=flag_gems.device)
    if bad_output == 0:
        out1 = torch.empty(2, dtype=torch.bool, device=flag_gems.device)
    else:
        out0 = torch.empty_like(inp)
    with pytest.raises(RuntimeError):
        flag_gems.fake_quantize_per_tensor_affine_cachemask_out(
            inp, 0.1, 0, 0, 255, out0=out0, out1=out1
        )
