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


_INT_SHAPES = [(128,), (64, 128), (32, 64, 128), (16, 32, 64, 128)]
_ARRAY_CASES = [
    ((128,), [32]),
    ((128,), [16, 48]),
    ((128,), [20, 40, 80]),
    ((64, 128), [32]),
    ((64, 128), [16, 48]),
    ((64, 128), [20, 40, 80]),
    ((32, 64, 128), [32]),
    ((32, 64, 128), [16, 48]),
    ((32, 64, 128), [20, 40, 80]),
    ((6, 8), [3, 3, 20]),
    ((6, 8), [5, 2]),
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
    split_dim = 0 if inp.ndim == 1 else 1
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


@pytest.mark.hsplit
@pytest.mark.parametrize("shape", _INT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("sections", [2, 4])
def test_hsplit_int(shape, dtype, sections):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp, True)
    reference = torch.hsplit(ref_inp, sections)

    result = flag_gems.hsplit(inp, sections)

    _assert_parts(result, reference, inp, ref_inp)


@pytest.mark.hsplit
@pytest.mark.parametrize("shape,indices", _ARRAY_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hsplit_array(shape, indices, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp, True)
    reference = torch.hsplit(ref_inp, indices)

    result = flag_gems.hsplit(inp, indices)

    _assert_parts(result, reference, inp, ref_inp)


@pytest.mark.hsplit
@pytest.mark.parametrize("indices_or_sections", [2, [3, 9]])
def test_hsplit_keyword_argument(indices_or_sections):
    inp = _make_input((4, 12), torch.float32)
    ref_inp = utils.to_reference(inp, True)
    reference = torch.hsplit(ref_inp, indices_or_sections)

    result = flag_gems.hsplit(
        inp, indices_or_sections=indices_or_sections
    )

    _assert_parts(result, reference, inp, ref_inp)


@pytest.mark.hsplit
def test_hsplit_exact_dtype():
    inp = _make_input((8, 12), torch.int8)
    ref_inp = utils.to_reference(inp)
    reference = torch.hsplit(ref_inp, [3, 9])

    result = flag_gems.hsplit(inp, [3, 9])

    _assert_parts(result, reference, inp, ref_inp)


@pytest.mark.hsplit
def test_hsplit_noncontiguous():
    inp = _make_input((8, 12), torch.float32).transpose(0, 1)
    ref_inp = utils.to_reference(inp, True)
    reference = torch.hsplit(ref_inp, [2, 6])

    result = flag_gems.hsplit(inp, [2, 6])

    _assert_parts(result, reference, inp, ref_inp)


@pytest.mark.hsplit
def test_hsplit_write_through_view():
    inp = _make_input((4, 8), torch.float32)
    expected = inp.clone()
    result = flag_gems.hsplit(inp, [2, 6])

    result[1].fill_(17.0)
    expected[:, 2:6].fill_(17.0)

    utils.gems_assert_equal(inp, expected)


@pytest.mark.hsplit
def test_hsplit_backward():
    inp = _make_input((4, 12), torch.float32, requires_grad=True)
    ref_inp = utils.to_reference(inp.detach().clone(), True).requires_grad_(True)
    reference = torch.hsplit(ref_inp, [3, 9])

    result = flag_gems.hsplit(inp, [3, 9])

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


@pytest.mark.hsplit
@pytest.mark.parametrize("indices_or_sections", [2, [3, 9]])
def test_hsplit_dispatch_registration(indices_or_sections):
    inp = _make_input((4, 12), torch.float32)
    ref_inp = utils.to_reference(inp, True)
    reference = torch.hsplit(ref_inp, indices_or_sections)

    with flag_gems.use_gems():
        result = torch.hsplit(inp, indices_or_sections)

    _assert_parts(result, reference, inp, ref_inp)


@pytest.mark.hsplit
@pytest.mark.parametrize("sections", [0, -1])
def test_hsplit_rejects_nonpositive_sections(sections):
    inp = torch.empty((4, 8), device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(RuntimeError):
        torch.hsplit(ref_inp, sections)
    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.hsplit(inp, sections)
    with pytest.raises((RuntimeError, ValueError)):
        with flag_gems.use_gems():
            torch.hsplit(inp, sections)


@pytest.mark.hsplit
def test_hsplit_requires_rank_one():
    inp = torch.empty((), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.hsplit(inp, 1)


@pytest.mark.hsplit
def test_hsplit_rejects_nondivisible_sections():
    inp = torch.empty((4, 7), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.hsplit(inp, 2)
