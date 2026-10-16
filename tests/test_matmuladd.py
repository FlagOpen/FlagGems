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
from .conftest import QUICK_MODE


if QUICK_MODE:
    _MNK_SHAPES = [(1, 1, 32)]
    _DTYPES = [torch.float32]
else:
    _MNK_SHAPES = [(1, 1, 32), (15, 160, 1024), (495, 5333, 71)]
    _DTYPES = utils.FLOAT_DTYPES


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


def _assert_unchanged(tensor, before, pointer):
    utils.gems_assert_equal(tensor, before)
    assert tensor.data_ptr() == pointer


def _run_case(input_tensor, other, bias, dtype, reduce_dim):
    ref_input = utils.to_reference(input_tensor, True)
    ref_other = utils.to_reference(other, True)
    ref_bias = utils.to_reference(bias, True)
    expected = torch.matmul(ref_input, ref_other) + ref_bias

    snapshots = [tensor.clone() for tensor in (input_tensor, other, bias)]
    pointers = [tensor.data_ptr() for tensor in (input_tensor, other, bias)]
    with flag_gems.use_gems():
        result = flag_gems.matmuladd(input_tensor, other, bias)

    assert result.shape == expected.shape
    assert result.dtype == input_tensor.dtype
    assert result.device == input_tensor.device
    result_storage = result.untyped_storage().data_ptr()
    assert result_storage not in {
        input_tensor.untyped_storage().data_ptr(),
        other.untyped_storage().data_ptr(),
        bias.untyped_storage().data_ptr(),
    }
    utils.gems_assert_close(result, expected, dtype, reduce_dim=reduce_dim)
    for tensor, before, pointer in zip(
        (input_tensor, other, bias), snapshots, pointers
    ):
        _assert_unchanged(tensor, before, pointer)


@pytest.mark.matmuladd
@pytest.mark.parametrize("M,N,K", _MNK_SHAPES)
@pytest.mark.parametrize("bias_kind", ["vector", "matrix"])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_matmuladd(M, N, K, bias_kind, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip(
            reason=(
                "Issue #3794: TX81 does not support the fp32 dot operation "
                "required by matmuladd"
            )
        )
    input_tensor = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    other = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias_shape = (N,) if bias_kind == "vector" else (M, N)
    bias = torch.randn(bias_shape, dtype=dtype, device=flag_gems.device)
    _run_case(input_tensor, other, bias, dtype, K)


@pytest.mark.matmuladd
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matmuladd_noncontiguous(dtype):
    input_tensor = torch.randn((4, 10), dtype=dtype, device=flag_gems.device)[:, ::2]
    other = torch.randn((7, 5), dtype=dtype, device=flag_gems.device).t()
    bias = torch.randn((4, 14), dtype=dtype, device=flag_gems.device)[:, ::2]
    assert not input_tensor.is_contiguous()
    assert not other.is_contiguous()
    assert not bias.is_contiguous()
    _run_case(input_tensor, other, bias, dtype, 5)


@pytest.mark.matmuladd
def test_matmuladd_is_not_autograd_registered():
    input_tensor = torch.randn(
        (3, 4), device=flag_gems.device, requires_grad=True
    )
    other = torch.randn((4, 5), device=flag_gems.device, requires_grad=True)
    bias = torch.randn((5,), device=flag_gems.device, requires_grad=True)
    result = flag_gems.matmuladd(input_tensor, other, bias)
    assert result.requires_grad is False


@pytest.mark.matmuladd
def test_matmuladd_rejects_incompatible_matmul_shapes():
    input_tensor = torch.randn((3, 4), device=flag_gems.device)
    other = torch.randn((6, 5), device=flag_gems.device)
    bias = torch.randn((5,), device=flag_gems.device)
    with pytest.raises(AssertionError, match="Incompatible dimensions"):
        flag_gems.matmuladd(input_tensor, other, bias)


@pytest.mark.matmuladd
def test_matmuladd_rejects_incompatible_bias_shape():
    input_tensor = torch.randn((3, 4), device=flag_gems.device)
    other = torch.randn((4, 5), device=flag_gems.device)
    bias = torch.randn((6,), device=flag_gems.device)
    with pytest.raises(AssertionError, match="Incompatible input shape"):
        flag_gems.matmuladd(input_tensor, other, bias)
