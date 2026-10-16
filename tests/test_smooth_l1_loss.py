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


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


_SHAPES = [(0,), (1,), (2, 3), (32, 17), (4, 8, 16), (2, 3, 16, 16)]


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_smooth_l1_loss(shape, dtype, reduction, beta):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if len(shape) >= 2 and shape[0] > 0:
        inp = inp.transpose(0, 1).contiguous().transpose(0, 1)
    inp_before = inp.clone()
    target_before = target.clone()
    reference = torch.ops.aten.smooth_l1_loss(
        utils.to_reference(inp, True),
        utils.to_reference(target, True),
        reduction,
        beta,
    )

    with flag_gems.use_gems():
        result = torch.ops.aten.smooth_l1_loss(inp, target, reduction, beta)

    utils.gems_assert_close(result, reference, dtype, equal_nan=True, atol=2e-2)
    utils.gems_assert_equal(inp, inp_before)
    utils.gems_assert_equal(target, target_before)


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize(
    "shape,target_shape",
    [
        ((0,), (0,)),
        ((1,), (1,)),
        ((2, 3), (2, 3)),
        ((32, 17), (32, 17)),
        ((4, 8, 16), (4, 8, 16)),
        ((2, 3, 16, 16), (2, 3, 16, 16)),
        ((2, 3, 4), (4,)),
        ((2, 1, 4), (3, 4)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_smooth_l1_loss_backward(
    shape, target_shape, dtype, reduction, beta
):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(
        target_shape, dtype=dtype, device=flag_gems.device
    )
    if beta == 0.0:
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
        target = torch.ones(
            target_shape, dtype=dtype, device=flag_gems.device
        )
    out_shape = torch.broadcast_shapes(shape, target_shape)
    if len(shape) >= 2 and shape[0] > 0:
        inp = inp.transpose(0, 1).contiguous().transpose(0, 1)
    if reduction == 0:
        grad_output = torch.randn(
            out_shape, dtype=dtype, device=flag_gems.device
        )
        if len(out_shape) >= 2 and out_shape[0] > 0:
            grad_output = (
                grad_output.transpose(0, 1).contiguous().transpose(0, 1)
            )
    else:
        grad_output = torch.randn(
            (), dtype=dtype, device=flag_gems.device
        )
    reference = torch.ops.aten.smooth_l1_loss_backward(
        utils.to_reference(grad_output, False).to(torch.float32),
        utils.to_reference(inp, False).to(torch.float32),
        utils.to_reference(target, False).to(torch.float32),
        reduction,
        beta,
    ).to(dtype)

    with flag_gems.use_gems():
        result = torch.ops.aten.smooth_l1_loss_backward(
            grad_output, inp, target, reduction, beta
        )

    utils.gems_assert_close(
        result, reference, dtype, equal_nan=True, atol=2e-2
    )


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_smooth_l1_loss_backward_scalar_grad_output():
    inp = torch.tensor([-1.0, -0.5, 1.0], device=flag_gems.device)
    target = torch.zeros_like(inp)
    grad_output = torch.tensor(2.0, device=flag_gems.device)
    reference = torch.ops.aten.smooth_l1_loss_backward(
        utils.to_reference(grad_output, False).to(torch.float32),
        utils.to_reference(inp, False).to(torch.float32),
        utils.to_reference(target, False).to(torch.float32),
        0,
        0.0,
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.smooth_l1_loss_backward(
            grad_output, inp, target, 0, 0.0
        )
    utils.gems_assert_close(result, reference, torch.float32)


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_smooth_l1_loss_backward_beta_zero_equal_inputs_cuda_behavior():
    if cfg.TO_CPU:
        pytest.skip("PyTorch CPU and CUDA differ for beta=0 with equal inputs.")
    inp = torch.zeros((4,), device=flag_gems.device)
    target = torch.zeros_like(inp)
    grad_output = torch.ones_like(inp)
    reference = torch.ops.aten.smooth_l1_loss_backward(
        grad_output, inp, target, 0, 0.0
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.smooth_l1_loss_backward(
            grad_output, inp, target, 0, 0.0
        )
    utils.gems_assert_close(
        result, reference, torch.float32, equal_nan=True
    )


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_smooth_l1_loss_broadcast(dtype, reduction):
    torch.manual_seed(0)
    inp = torch.randn((2, 3, 4), dtype=dtype, device=flag_gems.device)
    target = torch.randn((4,), dtype=dtype, device=flag_gems.device)
    reference = torch.ops.aten.smooth_l1_loss(
        utils.to_reference(inp, True), utils.to_reference(target, True), reduction, 1.0
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.smooth_l1_loss(inp, target, reduction, 1.0)
    utils.gems_assert_close(result, reference, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_special_values(dtype):
    inp = torch.tensor(
        [0.0, -0.0, 1.0, -2.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    target = torch.tensor(
        [0.0, 1.0, -1.0, -2.0, 1.0, float("-inf"), 0.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    reference = torch.ops.aten.smooth_l1_loss(
        utils.to_reference(inp, True), utils.to_reference(target, True), 0, 1.0
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.smooth_l1_loss(inp, target, 0, 1.0)
    utils.gems_assert_close(result, reference, dtype, equal_nan=True, atol=2e-2)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_smooth_l1_loss_out_preserves_storage(reduction):
    torch.manual_seed(0)
    inp = torch.randn((8, 16), device=flag_gems.device)
    target = torch.randn_like(inp)
    reference = torch.ops.aten.smooth_l1_loss(
        utils.to_reference(inp, True),
        utils.to_reference(target, True),
        reduction,
        0.5,
    )
    out_shape = inp.shape if reduction == 0 else ()
    out = torch.empty(out_shape, device=flag_gems.device)
    pointer = out.data_ptr()
    with flag_gems.use_gems():
        result = torch.ops.aten.smooth_l1_loss.out(
            inp, target, reduction, 0.5, out=out
        )
    assert result is out
    assert out.data_ptr() == pointer
    utils.gems_assert_close(out, reference, torch.float32)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_smooth_l1_loss_out_identity_and_resize(reduction):
    torch.manual_seed(0)
    inp = torch.randn((8, 16), device=flag_gems.device)
    target = torch.randn_like(inp)
    out = torch.empty((1,), device=flag_gems.device)
    reference = torch.ops.aten.smooth_l1_loss(
        utils.to_reference(inp, True), utils.to_reference(target, True), reduction, 0.5
    )
    with flag_gems.use_gems():
        result = torch.ops.aten.smooth_l1_loss.out(
            inp, target, reduction, 0.5, out=out
        )
    assert result is out
    assert out.shape == reference.shape
    utils.gems_assert_close(out, reference, torch.float32)


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss_functional_and_autograd():
    torch.manual_seed(0)
    inp = torch.randn((8, 16), device=flag_gems.device, requires_grad=True)
    target = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp, True).requires_grad_(True)
    ref_target = utils.to_reference(target, True)
    reference = torch.nn.functional.smooth_l1_loss(
        ref_inp, ref_target, reduction="mean", beta=0.5
    )
    with flag_gems.use_gems():
        result = torch.nn.functional.smooth_l1_loss(
            inp, target, reduction="mean", beta=0.5
        )
    (reference_grad,) = torch.autograd.grad(reference, ref_inp)
    (result_grad,) = torch.autograd.grad(result, inp)
    utils.gems_assert_close(result, reference, torch.float32)
    utils.gems_assert_close(result_grad, reference_grad, torch.float32)


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss_rejects_negative_beta():
    inp = torch.randn((8,), device=flag_gems.device)
    target = torch.randn_like(inp)
    with pytest.raises(RuntimeError, match="negative"):
        with flag_gems.use_gems():
            torch.ops.aten.smooth_l1_loss(inp, target, 1, -1.0)


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_smooth_l1_loss_backward_rejects_negative_beta():
    grad_output = torch.randn((), device=flag_gems.device)
    inp = torch.randn((8,), device=flag_gems.device)
    target = torch.randn_like(inp)
    with pytest.raises(RuntimeError, match="negative"):
        with flag_gems.use_gems():
            torch.ops.aten.smooth_l1_loss_backward(
                grad_output, inp, target, 1, -1.0
            )
