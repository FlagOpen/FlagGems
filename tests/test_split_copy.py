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

if cfg.QUICK_MODE:
    # (input_shape, dim, split_size) triples.
    SPLIT_COPY_CASES = [
        ((8, 6), 0, 4),
        ((10, 8), 0, 4),
        ((10, 8), 1, 3),
    ]
else:
    # (input_shape, dim, split_size) triples. Divisible and non-divisible
    # splits (the last chunk is smaller), every dim position including
    # negative dims, 1D/3D/4D inputs, empty tensors and split_size larger
    # than the dimension are covered.
    SPLIT_COPY_CASES = [
        ((8, 6), 0, 4),
        ((10, 8), 0, 4),
        ((10, 8), 0, 3),
        ((10, 8), 0, 16),
        ((10, 8), 1, 3),
        ((10, 8), 1, 5),
        ((10, 8), 1, 8),
        ((10, 8), -2, 4),
        ((10, 8), -1, 3),
        ((10,), 0, 4),
        ((10,), 0, 15),
        ((2, 6, 5), 1, 4),
        ((2, 5, 6), -1, 4),
        ((2, 4, 5, 7), 2, 3),
        ((2, 4, 5, 7), -3, 2),
        ((3,), 0, 2),
        ((1, 1), 1, 1),
        ((0, 4), 0, 2),
        ((4, 0), 1, 2),
    ]


def _assert_chunks_match(res_out, ref_out, ref_inp, dtype):
    assert len(res_out) == len(ref_out), "Number of splits mismatch"
    for res, ref in zip(res_out, ref_out):
        assert list(res.shape) == list(ref.shape)
        assert res.is_contiguous()
        assert not torch._C._is_alias_of(res, ref_inp)
        utils.gems_assert_equal(res, ref)


@pytest.mark.split_copy
@pytest.mark.parametrize("input_shape, dim, split_size", SPLIT_COPY_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_split_copy(input_shape, dim, split_size, dtype):
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.split_copy(ref_inp, split_size, dim=dim)
    res_out = flag_gems.split_copy(inp, split_size, dim=dim)

    _assert_chunks_match(res_out, ref_out, inp, dtype)


@pytest.mark.split_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_split_copy_noncontiguous_input(dtype):
    base = torch.randn(6, 4, dtype=dtype, device=flag_gems.device)
    inp = base.t()
    ref_inp = utils.to_reference(inp)

    for dim, split_size in [(0, 2), (0, 3), (1, 2), (1, 1), (-2, 2)]:
        ref_out = torch.ops.aten.split_copy(ref_inp, split_size, dim=dim)
        res_out = flag_gems.split_copy(inp, split_size, dim=dim)
        _assert_chunks_match(res_out, ref_out, inp, dtype)


@pytest.mark.split_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_split_copy_strided_input(dtype):
    # A stride-2 view: the strided gather path must honor the real strides.
    base = torch.randn(4, 8, 6, dtype=dtype, device=flag_gems.device)
    inp = base[:, ::2, :]
    ref_inp = utils.to_reference(inp)

    for dim, split_size in [(0, 3), (1, 2), (2, 4), (-1, 5)]:
        ref_out = torch.ops.aten.split_copy(ref_inp, split_size, dim=dim)
        res_out = flag_gems.split_copy(inp, split_size, dim=dim)
        _assert_chunks_match(res_out, ref_out, inp, dtype)


@pytest.mark.split_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_split_copy_broadcast_input(dtype):
    # An expanded (zero-stride) input; the copies must materialize the values.
    base = torch.randn(1, 6, dtype=dtype, device=flag_gems.device)
    inp = base.expand(4, 6)
    ref_inp = utils.to_reference(inp)

    for dim, split_size in [(0, 2), (1, 4)]:
        ref_out = torch.ops.aten.split_copy(ref_inp, split_size, dim=dim)
        res_out = flag_gems.split_copy(inp, split_size, dim=dim)
        _assert_chunks_match(res_out, ref_out, inp, dtype)


@pytest.mark.split_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_split_copy_storage_offset(dtype):
    base = torch.randn(20, dtype=dtype, device=flag_gems.device)
    inp = base[4:12].reshape(2, 4)
    ref_inp = utils.to_reference(inp)

    for dim, split_size in [(0, 1), (1, 2)]:
        ref_out = torch.ops.aten.split_copy(ref_inp, split_size, dim=dim)
        res_out = flag_gems.split_copy(inp, split_size, dim=dim)
        _assert_chunks_match(res_out, ref_out, inp, dtype)


@pytest.mark.split_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_split_copy_independent_storage(dtype):
    inp = torch.randn(10, 6, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.split_copy(inp, 4, dim=1)
    ref = [utils.to_reference(chunk.clone()) for chunk in res_out]

    inp.add_(1.0)
    # The copies must not change when the source is mutated in place.
    for chunk, chunk_ref in zip(res_out, ref):
        utils.gems_assert_equal(chunk, chunk_ref)


@pytest.mark.split_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_split_copy_tail_chunk(dtype):
    # The non-divisible tail chunk must carry the leftover extent exactly.
    inp = torch.randn(13, 10, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim, split_size in [(0, 4), (0, 5), (1, 3)]:
        ref_out = torch.ops.aten.split_copy(ref_inp, split_size, dim=dim)
        res_out = flag_gems.split_copy(inp, split_size, dim=dim)
        assert len(res_out) == len(ref_out)
        if dim == 0:
            assert tuple(res_out[-1].shape) == (13 % split_size, 10)
        else:
            assert tuple(res_out[-1].shape) == (13, 10 % split_size)
        _assert_chunks_match(res_out, ref_out, inp, dtype)


@pytest.mark.split_copy
def test_accuracy_split_copy_error_paths():
    inp = torch.randn(10, 8, device=flag_gems.device)

    with pytest.raises((RuntimeError, IndexError)):
        torch.ops.aten.split_copy(inp, 0, dim=0)
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems.split_copy(inp, 0, dim=0)

    with pytest.raises((RuntimeError, IndexError)):
        torch.ops.aten.split_copy(inp, -1, dim=0)
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems.split_copy(inp, -1, dim=0)

    with pytest.raises((RuntimeError, IndexError)):
        torch.ops.aten.split_copy(inp, 3, dim=5)
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems.split_copy(inp, 3, dim=5)

    with pytest.raises((RuntimeError, IndexError)):
        torch.ops.aten.split_copy(torch.tensor(1.0, device=flag_gems.device), 1, dim=0)
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems.split_copy(torch.tensor(1.0, device=flag_gems.device), 1, dim=0)
