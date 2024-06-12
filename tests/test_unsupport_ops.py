import torch
import pytest
import flag_gems
from .accuracy_utils import *


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cumsum(shape, dtype):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, False)

    ref_out = torch.cumsum(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.cumsum(inp, dim=dim)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.parametrize(
    "N, C, H, W, num_groups",
    [
        (16, 3, 16, 16, 1),
        (32, 32, 32, 32, 8),
        (1, 32, 32, 32, 8),
        (1, 32, 32, 32, 16),
        (1, 64, 32, 32, 16),
        (1, 64, 32, 32, 32),
        (1, 64, 32, 32, 64),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_groupnorm(N, C, H, W, num_groups, dtype):
    HW = H * W
    inp = torch.randn(size=(N, C, H, W), dtype=dtype, device="musa", requires_grad=True)
    weight = torch.randn(size=(C,), dtype=dtype, device="musa", requires_grad=True)
    bias = torch.randn(size=(C,), dtype=dtype, device="musa", requires_grad=True)
    eps = 1e-5

    ref_inp = to_reference(inp, False)
    ref_weight = to_reference(weight, False)
    ref_bias = to_reference(bias, False)

    ref_out = torch.nn.functional.group_norm(
        ref_inp, num_groups, weight=ref_weight, bias=ref_bias, eps=eps
    )
    ref_mean = torch.mean(ref_inp.reshape([N, num_groups, -1]), dim=2)
    ref_var = torch.var(ref_inp.reshape([N, num_groups, -1]), dim=2, correction=0)
    ref_rstd = torch.rsqrt(ref_var + eps)

    (res_out, res_mean, res_rstd) = flag_gems.group_norm(
        inp, weight, bias, N, C, HW, num_groups, eps
    )

    gems_assert_close(res_mean, ref_mean, dtype)
    gems_assert_close(res_rstd, ref_rstd, dtype)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, False)

    (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
    )
    (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
        res_out, (inp, weight, bias), out_grad
    )
    group_size = C // num_groups
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=group_size * HW)
    gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N * HW)
    gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=N * HW)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_layernorm(shape, dtype):
    M = shape[0]
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape, dtype=dtype, device="musa", requires_grad=True)
    weight = torch.randn(layer_shape, dtype=dtype, device="musa", requires_grad=True)
    bias = torch.randn(layer_shape, dtype=dtype, device="musa", requires_grad=True)
    eps = 1e-5

    ref_inp = to_reference(inp, False)
    ref_weight = to_reference(weight, False)
    ref_bias = to_reference(bias, False)

    ref_out = torch.layer_norm(
        ref_inp,
        list(layer_shape),
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )
    (res_out, res_mean, res_rstd) = flag_gems.layer_norm(
        inp, list(layer_shape), weight=weight, bias=bias, eps=eps
    )

    ref_mean = torch.mean(ref_inp, dim=1)
    ref_var = torch.var(ref_inp, dim=1, correction=0)
    ref_rstd = torch.rsqrt(ref_var + eps)
    gems_assert_close(res_mean, ref_mean, dtype)
    gems_assert_close(res_rstd, ref_rstd, dtype)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, False)

    (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
    )
    (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
        res_out, (inp, weight, bias), out_grad
    )
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=N)
    gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=M)
    gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=M)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_varmean(shape, dim, correction, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, False)

    ref_var, ref_mean = torch.var_mean(
        ref_inp, dim, correction=correction, keepdim=keepdim
    )
    with flag_gems.use_gems():
        res_var, res_mean = torch.var_mean(
            inp, dim, correction=correction, keepdim=keepdim
        )

    gems_assert_close(res_mean, ref_mean, dtype)
    gems_assert_close(res_var, ref_var, dtype)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("ord", [2, float("inf"), -float("inf"), 0, 1])
@pytest.mark.parametrize("dim", DIMS_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_vectornorm(shape, ord, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp = to_reference(inp, False)

    ref_out = torch.linalg.vector_norm(ref_inp, ord, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.linalg.vector_norm(inp, ord, dim, keepdim)

    gems_assert_close(res_out, ref_out, dtype)

