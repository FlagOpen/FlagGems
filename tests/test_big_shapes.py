import pytest
import torch

import flag_gems

from .accuracy_utils import (
    DIM_LIST,
    DIMS_LIST,
    FLOAT_DTYPES,
    BIG_REDUCTION_SHAPES,
    DEVICE,
    gems_assert_close,
    gems_assert_equal,
    skip_expr,
    skip_reason,
    to_reference,
)

@pytest.mark.parametrize("shape", BIG_REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cumsum(shape, dtype):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    ref_inp = to_reference(inp, True)

    ref_out = torch.cumsum(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.cumsum(inp, dim=dim)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])

@pytest.mark.parametrize("shape", BIG_REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1])
def test_accuracy_log_softmax(shape, dtype, dim):
    inp = torch.randn(shape, dtype=dtype, device=DEVICE, requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.log_softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.log_softmax(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])

@pytest.mark.parametrize("shape", BIG_REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1])
def test_accuracy_softmax(shape, dtype, dim):
    inp = torch.randn(shape, dtype=dtype, device=DEVICE, requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.softmax(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


