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


FLOAT_DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES


@pytest.fixture(autouse=True)
def _seed_random():
    torch.manual_seed(0)


def _total_weight(target, weight, ignore_index, dtype):
    valid = target != ignore_index
    if weight is None:
        return valid.sum().to(dtype=dtype)
    valid_targets = target[valid]
    if valid_targets.numel() == 0:
        return torch.zeros((), dtype=dtype, device=target.device)
    return weight[valid_targets].sum()


def _make_case(shape, dtype, reduction, use_weight, ignore_index):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    class_count = shape[-1]
    target_shape = () if len(shape) == 1 else shape[:-1]
    target = torch.randint(
        0, class_count, target_shape, device=flag_gems.device
    )
    if (
        target.ndim > 0
        and target.numel() > 1
        and 0 <= ignore_index < class_count
    ):
        target.reshape(-1)[0] = ignore_index
    weight = (
        torch.rand(class_count, dtype=dtype, device=flag_gems.device) + 0.25
        if use_weight
        else None
    )
    grad_shape = target.shape if reduction == 0 else ()
    grad_output = torch.randn(
        grad_shape, dtype=dtype, device=flag_gems.device
    )
    total_weight = _total_weight(target, weight, ignore_index, dtype)
    return grad_output, inp, target, weight, total_weight


def _reference_args(grad_output, inp, target, weight, total_weight):
    return (
        utils.to_reference(grad_output, True),
        utils.to_reference(inp, True),
        utils.to_reference(target, False),
        None if weight is None else utils.to_reference(weight, True),
        utils.to_reference(total_weight, True),
    )


@pytest.mark.nll_loss_backward
@pytest.mark.parametrize("shape", [(17,), (8, 17), (31, 7)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("use_weight", [False, True])
@pytest.mark.parametrize("ignore_index", [-100, 1])
def test_nll_loss_backward(
    shape, dtype, reduction, use_weight, ignore_index
):
    grad_output, inp, target, weight, total_weight = _make_case(
        shape, dtype, reduction, use_weight, ignore_index
    )
    ref_grad, ref_inp, ref_target, ref_weight, ref_total_weight = (
        _reference_args(grad_output, inp, target, weight, total_weight)
    )
    reference = torch.ops.aten.nll_loss_backward(
        ref_grad,
        ref_inp,
        ref_target,
        ref_weight,
        reduction,
        ignore_index,
        ref_total_weight,
    )

    result = flag_gems.nll_loss_backward(
        grad_output,
        inp,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight,
    )

    assert result.shape == inp.shape
    assert result.dtype == dtype
    utils.gems_assert_close(
        result, reference, dtype, reduce_dim=shape[-1]
    )


@pytest.mark.nll_loss_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_nll_loss_backward_noncontiguous(dtype, reduction):
    torch.manual_seed(0)
    inp = torch.randn(
        (11, 5), dtype=dtype, device=flag_gems.device
    ).T
    target = torch.randint(0, inp.shape[-1], (inp.shape[0],), device=inp.device)
    target[0] = 1
    weight = torch.rand(
        inp.shape[-1], dtype=dtype, device=flag_gems.device
    )
    grad_shape = target.shape if reduction == 0 else ()
    grad_output = torch.randn(
        grad_shape, dtype=dtype, device=flag_gems.device
    )
    total_weight = _total_weight(target, weight, 1, dtype)
    ref_grad, ref_inp, ref_target, ref_weight, ref_total_weight = (
        _reference_args(grad_output, inp, target, weight, total_weight)
    )
    reference = torch.ops.aten.nll_loss_backward(
        ref_grad,
        ref_inp,
        ref_target,
        ref_weight,
        reduction,
        1,
        ref_total_weight,
    )

    result = flag_gems.nll_loss_backward(
        grad_output,
        inp,
        target,
        weight,
        reduction,
        1,
        total_weight,
    )

    utils.gems_assert_close(
        result, reference, dtype, reduce_dim=inp.shape[-1]
    )


@pytest.mark.nll_loss_backward
def test_nll_loss_backward_all_ignored_mean():
    inp = torch.randn((4, 7), device=flag_gems.device)
    target = torch.ones((4,), device=inp.device, dtype=torch.int64)
    grad_output = torch.randn((), device=inp.device)
    total_weight = torch.zeros((), device=inp.device)
    ref_grad, ref_inp, ref_target, _, ref_total_weight = _reference_args(
        grad_output, inp, target, None, total_weight
    )
    reference = torch.ops.aten.nll_loss_backward(
        ref_grad, ref_inp, ref_target, None, 1, 1, ref_total_weight
    )

    result = flag_gems.nll_loss_backward(
        grad_output, inp, target, None, 1, 1, total_weight
    )

    utils.gems_assert_close(
        result, reference, torch.float32, equal_nan=True
    )
