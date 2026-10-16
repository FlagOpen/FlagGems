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


_SHAPES = [(2, 3), (128, 256), (512, 512), (0, 3)]


def _call(inp, target, log_input, full, eps, reduction):
    return flag_gems.poisson_nll_loss(
        inp, target, log_input, full, eps, reduction
    )


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("log_input", [True, False])
@pytest.mark.parametrize("full", [False, True])
def test_poisson_nll_loss(shape, dtype, reduction, log_input, full):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if not log_input:
        inp = inp.abs() + 0.1
    target = torch.randint(0, 5, shape, device=flag_gems.device).to(dtype)
    reference = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(inp, True),
        utils.to_reference(target, True),
        log_input,
        full,
        1e-8,
        reduction,
    )
    result = _call(inp, target, log_input, full, 1e-8, reduction)
    utils.gems_assert_close(
        result,
        reference,
        dtype,
        equal_nan=True,
        reduce_dim=max(inp.numel(), 1) if reduction in (1, 2) else 1,
    )


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("reduction", [0, 1, 2, 3])
def test_poisson_nll_loss_broadcast_noncontiguous(reduction):
    torch.manual_seed(0)
    inp = torch.randn((7, 5), device=flag_gems.device).T
    target = torch.randint(0, 5, (1, 7), device=flag_gems.device).float()
    reference = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(inp, True),
        utils.to_reference(target, True),
        True,
        True,
        1e-8,
        reduction,
    )
    result = _call(inp, target, True, True, 1e-8, reduction)
    utils.gems_assert_close(result, reference, torch.float32, equal_nan=True)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_poisson_nll_loss_boundary(dtype):
    inp = torch.tensor(
        [0.0, -0.0, 1e-5, float("inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    target = torch.tensor(
        [1.0, 0.0, 2.0, 0.0, 1.0], dtype=dtype, device=flag_gems.device
    )
    reference = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(inp, True),
        utils.to_reference(target, True),
        False,
        True,
        1e-8,
        0,
    )
    result = _call(inp, target, False, True, 1e-8, 0)
    utils.gems_assert_close(result, reference, dtype, equal_nan=True)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize(
    "input_dtype,target_dtype,expected_dtype",
    [
        (torch.int64, torch.float16, torch.float32),
        (torch.float16, torch.int64, torch.float16),
        (torch.bool, torch.float32, torch.float32),
        (torch.float32, torch.bool, torch.float32),
        (torch.float32, torch.float64, torch.float64),
    ],
)
def test_poisson_nll_loss_dtype_promotion(
    input_dtype, target_dtype, expected_dtype
):
    inp = torch.ones((2, 3), dtype=input_dtype, device=flag_gems.device)
    target = torch.ones((1, 3), dtype=target_dtype, device=flag_gems.device)
    reference = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(inp, False),
        utils.to_reference(target, False),
        True,
        False,
        1e-8,
        0,
    )
    result = _call(inp, target, True, False, 1e-8, 0)
    assert result.dtype == expected_dtype
    assert reference.dtype == expected_dtype
    utils.gems_assert_close(result, reference, expected_dtype)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("eps", [0.0, -1e-3, float("inf"), float("nan")])
def test_poisson_nll_loss_special_eps(eps):
    inp = torch.tensor([0.0, 0.5, 2.0], device=flag_gems.device)
    target = torch.tensor([1.0, 2.0, 3.0], device=flag_gems.device)
    reference = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(inp, True),
        utils.to_reference(target, True),
        False,
        False,
        eps,
        0,
    )
    result = _call(inp, target, False, False, eps, 0)
    utils.gems_assert_close(result, reference, torch.float32, equal_nan=True)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("log_input", [False, True])
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize(
    "case", ["broadcast_conjugate", "large", "empty", "real_input", "real_target"]
)
def test_poisson_nll_loss_complex(dtype, log_input, reduction, case):
    if dtype == torch.complex128 and not utils.fp64_is_supported:
        pytest.skip("FP64 is not supported")
    torch.manual_seed(0)
    shape = (65537,) if case == "large" else (0, 3) if case == "empty" else (3, 7)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if case == "broadcast_conjugate":
        inp = inp.T.conj()
        target = target[:1].T.conj()
    elif case == "real_input":
        inp = inp.real.abs() + 0.5
    elif case == "real_target":
        target = target.real.abs() + 0.5
    full = case == "real_target"
    reference = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(inp, False),
        utils.to_reference(target, False),
        log_input,
        full,
        1e-8,
        reduction,
    )
    result = _call(inp, target, log_input, full, 1e-8, reduction)
    assert result.dtype == reference.dtype
    utils.gems_assert_close(
        result,
        reference,
        dtype,
        equal_nan=True,
        reduce_dim=max(inp.numel(), 1) if reduction in (1, 2) else 1,
    )


@pytest.mark.poisson_nll_loss
def test_poisson_nll_loss_complex_target_full_rejected():
    inp = torch.ones(3, dtype=torch.complex64, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.poisson_nll_loss(
            utils.to_reference(inp, False),
            utils.to_reference(inp, False),
            True,
            True,
            1e-8,
            0,
        )
    with pytest.raises(RuntimeError):
        _call(inp, inp, True, True, 1e-8, 0)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("size", [65536, 65537])
@pytest.mark.parametrize("reduction", [1, 2])
def test_poisson_nll_loss_reduction_boundary(size, reduction):
    torch.manual_seed(0)
    inp = torch.randn(size, device=flag_gems.device)
    target = torch.randint(0, 5, (size,), device=flag_gems.device).float()
    reference = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(inp, True),
        utils.to_reference(target, True),
        True,
        False,
        1e-8,
        reduction,
    )
    result = _call(inp, target, True, False, 1e-8, reduction)
    utils.gems_assert_close(
        result, reference, torch.float32, reduce_dim=size, equal_nan=True
    )


@pytest.mark.poisson_nll_loss
def test_poisson_nll_loss_autograd():
    torch.manual_seed(0)
    inp = torch.randn((3, 5), device=flag_gems.device, requires_grad=True)
    target = torch.rand_like(inp)
    ref_inp = utils.to_reference(inp, True).requires_grad_(True)
    ref_target = utils.to_reference(target, True)
    reference = torch.ops.aten.poisson_nll_loss(
        ref_inp, ref_target, True, False, 1e-8, 1
    )
    result = _call(inp, target, True, False, 1e-8, 1)
    (reference_grad,) = torch.autograd.grad(reference, ref_inp)
    (result_grad,) = torch.autograd.grad(result, inp)
    utils.gems_assert_close(result_grad, reference_grad, torch.float32)


@pytest.mark.poisson_nll_loss
def test_poisson_nll_loss_rejects_invalid_inputs():
    inp = torch.ones((2, 3), device=flag_gems.device)
    invalid_shape = torch.ones((4, 3), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.poisson_nll_loss(
            utils.to_reference(inp, False),
            utils.to_reference(invalid_shape, False),
            True,
            False,
            1e-8,
            0,
        )
    with pytest.raises(RuntimeError):
        _call(inp, invalid_shape, True, False, 1e-8, 0)

    bool_input = torch.ones((2, 3), dtype=torch.bool, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.poisson_nll_loss(
            utils.to_reference(bool_input, False),
            utils.to_reference(bool_input, False),
            True,
            False,
            1e-8,
            0,
        )
    with pytest.raises(RuntimeError):
        _call(bool_input, bool_input, True, False, 1e-8, 0)

    with pytest.raises(RuntimeError):
        torch.ops.aten.poisson_nll_loss(
            utils.to_reference(inp, False),
            utils.to_reference(bool_input, False),
            True,
            True,
            1e-8,
            0,
        )
    with pytest.raises(RuntimeError):
        _call(inp, bool_input, True, True, 1e-8, 0)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_poisson_nll_loss_bool_input_nonlog_rejected(full, reduction):
    inp = torch.ones((2, 3), dtype=torch.bool, device=flag_gems.device)
    target = torch.ones((1, 3), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.poisson_nll_loss(
            utils.to_reference(inp, False),
            utils.to_reference(target, False),
            False,
            full,
            1e-8,
            reduction,
        )
    with pytest.raises(RuntimeError):
        _call(inp, target, False, full, 1e-8, reduction)
