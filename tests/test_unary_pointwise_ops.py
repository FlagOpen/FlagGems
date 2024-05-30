import torch
import pytest
import flag_gems
from .accuracy_utils import *


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_abs(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.abs(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.abs(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwisenot(shape, dtype):
    inp = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
    )
    ref_inp = to_reference(inp)

    ref_out = torch.bitwise_not(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.bitwise_not(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cos(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.cos(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cos(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.exp(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.exp(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_gelu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.gelu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.gelu(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_accuracy_gelu_and_mul(shape, approximate, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(
        torch.nn.functional.gelu(ref_inp1, approximate=approximate), ref_inp2
    )
    with flag_gems.use_gems():
        res_out = flag_gems.gelu_and_mul(inp1, inp2, approximate)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_isinf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    inp = torch.masked_fill(inp, inp > 1.0, -float("inf"))
    ref_inp = to_reference(inp)

    ref_out = torch.isinf(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.isinf(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_isnan(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    inp = torch.masked_fill(inp, inp > 1.0, float("nan"))
    ref_inp = to_reference(inp)

    ref_out = torch.isnan(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.isnan(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_neg(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.neg(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.neg(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_reciprocal(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.reciprocal(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.reciprocal(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_relu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.relu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.relu(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rsqrt(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.rsqrt(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.rsqrt(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sigmoid(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.sigmoid(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sigmoid(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_silu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.silu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_silu_and_mul(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(torch.nn.functional.silu(ref_inp1), ref_inp2)
    with flag_gems.use_gems():
        res_out = flag_gems.silu_and_mul(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sin(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.sin(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sin(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_tanh(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.tanh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.tanh(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("diagonal", [-3, -1, 0, 1, 3])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_triu(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.triu(ref_inp, diagonal)
    with flag_gems.use_gems():
        res_out = torch.triu(inp, diagonal)

    gems_assert_equal(res_out, ref_out)
