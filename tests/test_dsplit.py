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


_D_SPLIT_CASES = [
    ((4, 6, 8), 2),
    ((8, 4, 12), 4),
    ((12, 8, 16), 4),
    ((16, 16, 8), 2),
    ((20, 10, 15), 5),
    ((4, 6, 8), [2]),
    ((4, 6, 8), [4]),
    ((6, 8, 10), [2, 6]),
    ((8, 4, 12), [3, 6, 9]),
    ((10, 5, 15), [5, 10]),
    ((4, 6, 8, 3), 2),
    ((8, 4, 12, 5), 4),
    ((6, 8, 10, 7), [2, 6]),
    ((10, 5, 15, 3), [5, 10]),
    ((2, 3, 4), [2, 2, 10]),
    ((2, 3, 8), [3, 1]),
]


def _make_input(shape, dtype, *, requires_grad=False):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-128, 127, shape, dtype=dtype, device=flag_gems.device)
        if inp.numel():
            inp.flatten()[0] = -128
        if inp.numel() > 1:
            inp.flatten()[1] = 126
    return inp.requires_grad_(requires_grad)


def _assert_parts(result, reference, inp, ref_inp):
    assert type(result) is type(reference)
    assert len(result) == len(reference)
    storage_base = inp.untyped_storage().data_ptr()
    split_dim = 2
    for res_part, ref_part in zip(result, reference):
        assert res_part is not inp
        assert res_part.shape == ref_part.shape
        assert res_part.dtype == inp.dtype
        assert res_part.stride() == inp.stride()
        ref_delta = ref_part.storage_offset() - ref_inp.storage_offset()
        ref_stride = ref_inp.stride(split_dim)
        assert ref_delta % ref_stride == 0
        expected_offset = inp.storage_offset() + (
            ref_delta // ref_stride
        ) * inp.stride(split_dim)
        assert res_part.storage_offset() == expected_offset
        assert res_part._is_view()
        assert res_part.untyped_storage().data_ptr() == storage_base
        if inp.dtype.is_floating_point:
            utils.gems_assert_close(res_part, ref_part, inp.dtype)
        else:
            utils.gems_assert_equal(res_part, ref_part)


@pytest.mark.dsplit
@pytest.mark.parametrize("shape,indices_or_sections", _D_SPLIT_CASES)
def test_dsplit_float32(shape, indices_or_sections):
    inp = _make_input(shape, torch.float32)
    ref_inp = utils.to_reference(inp, True)
    reference = torch.dsplit(ref_inp, indices_or_sections)

    result = flag_gems.dsplit(inp, indices_or_sections)

    _assert_parts(result, reference, inp, ref_inp)


@pytest.mark.dsplit
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int8])
def test_dsplit_dtypes(dtype):
    inp = _make_input((4, 6, 8), dtype)
    ref_inp = (
        utils.to_reference(inp, True)
        if dtype.is_floating_point
        else utils.to_reference(inp)
    )
    reference = torch.dsplit(ref_inp, 2)

    result = flag_gems.dsplit(inp, 2)

    _assert_parts(result, reference, inp, ref_inp)


@pytest.mark.dsplit
@pytest.mark.parametrize("indices_or_sections", [2, [3, 6]])
def test_dsplit_keyword_argument(indices_or_sections):
    inp = _make_input((2, 3, 8), torch.float32)
    ref_inp = utils.to_reference(inp, True)
    reference = torch.dsplit(ref_inp, indices_or_sections)

    result = flag_gems.dsplit(
        inp, indices_or_sections=indices_or_sections
    )

    _assert_parts(result, reference, inp, ref_inp)


@pytest.mark.dsplit
def test_dsplit_noncontiguous():
    base = _make_input((4, 6, 8, 5), torch.float32)
    inp = base.transpose(1, 2)[:, :, :, ::2]
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp, True)
    reference = torch.dsplit(ref_inp, [2, 5])

    result = flag_gems.dsplit(inp, [2, 5])

    _assert_parts(result, reference, inp, ref_inp)


@pytest.mark.dsplit
def test_dsplit_write_through_view():
    inp = _make_input((2, 3, 8), torch.float32)
    expected = inp.clone()
    result = flag_gems.dsplit(inp, [2, 5])

    result[1].fill_(17.0)
    expected[:, :, 2:5].fill_(17.0)

    utils.gems_assert_equal(inp, expected)


@pytest.mark.dsplit
def test_dsplit_backward():
    inp = _make_input((3, 4, 8), torch.float32, requires_grad=True)
    ref_inp = utils.to_reference(inp.detach().clone(), True).requires_grad_(True)
    reference = torch.dsplit(ref_inp, [3, 6])

    result = flag_gems.dsplit(inp, [3, 6])

    upstream = tuple(
        torch.arange(part.numel(), dtype=part.dtype, device=part.device).reshape(
            part.shape
        )
        + (index + 1) * 0.25
        for index, part in enumerate(result)
    )
    ref_upstream = tuple(utils.to_reference(part, True) for part in upstream)
    result_grad = torch.autograd.grad(result, inp, grad_outputs=upstream)[0]
    reference_grad = torch.autograd.grad(
        reference, ref_inp, grad_outputs=ref_upstream
    )[0]
    utils.gems_assert_close(result_grad, reference_grad, torch.float32)


@pytest.mark.dsplit
@pytest.mark.parametrize("indices_or_sections", [2, [3, 6]])
def test_dsplit_dispatch_registration(indices_or_sections):
    inp = _make_input((2, 3, 8), torch.float32)
    ref_inp = utils.to_reference(inp, True)
    reference = torch.dsplit(ref_inp, indices_or_sections)

    with flag_gems.use_gems():
        result = torch.dsplit(inp, indices_or_sections)

    _assert_parts(result, reference, inp, ref_inp)


@pytest.mark.dsplit
@pytest.mark.parametrize("shape", [(), (8,), (4, 8)])
def test_dsplit_requires_rank_three(shape):
    inp = torch.empty(shape, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.dsplit(inp, 2)


@pytest.mark.dsplit
def test_dsplit_rejects_nondivisible_sections():
    inp = torch.empty((2, 3, 7), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.dsplit(inp, 3)


@pytest.mark.dsplit
@pytest.mark.parametrize("sections", [0, -1])
def test_dsplit_rejects_nonpositive_sections(sections):
    inp = torch.empty((2, 3, 8), device=flag_gems.device)

    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.dsplit(inp, sections)
