import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    REDUCTION_SHAPES,
    gems_assert_close,
    to_reference,
)
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES
DIMS_LIST = [1] if QUICK_MODE else [0, 1, [0, 1], [1, 0]]
KEEPDIM_DIMS = (
    [(True, DIMS_LIST[0])] if QUICK_MODE else list(zip([True, False] * 2, DIMS_LIST))
)


@pytest.mark.group_norm
@pytest.mark.native_group_norm
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
    inp = torch.randn(size=(N, C, H, W), dtype=dtype, device="cuda", requires_grad=True)
    weight = torch.randn(size=(C,), dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.randn(size=(C,), dtype=dtype, device="cuda", requires_grad=True)
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

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
    ref_grad = to_reference(out_grad, True)

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


@pytest.mark.layer_norm
@pytest.mark.native_layer_norm
@pytest.mark.parametrize(
    "shape",
    [(1, 40999)]
    if QUICK_MODE
    else [
        (200, 36),
        (4096, 100),
        (1, 40999),
        (100, 40499),
        (4096, 256),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_layernorm(shape, dtype):
    M = shape[0]
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape[:2], dtype=dtype, device="cuda", requires_grad=True)
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.randn(layer_shape, dtype=dtype, device="cuda", requires_grad=True)
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

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
    gems_assert_close(res_mean, ref_mean, res_mean.dtype)
    gems_assert_close(res_rstd, ref_rstd, res_rstd.dtype)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
    )
    (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
        res_out, (inp, weight, bias), out_grad
    )
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=N)
    gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=M)
    gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=M)


@pytest.mark.rms_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rmsnorm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape[:2], dtype=dtype, device="cuda")
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda")
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_weight = to_reference(weight, True)

    def _torch_rms_norm(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    ref_out = _torch_rms_norm(ref_inp, weight=ref_weight, eps=eps)

    res_out = flag_gems.rms_norm(inp, list(layer_shape), weight=weight, eps=eps)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skip_layer_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_skip_layernorm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape[:2], dtype=dtype, device="cuda")
    residual = torch.randn(shape[:2], dtype=dtype, device="cuda")
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda")
    bias = torch.randn(layer_shape, dtype=dtype, device="cuda")
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_residual = to_reference(residual, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

    ref_out = torch.layer_norm(
        ref_inp + ref_residual,
        list(layer_shape),
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )
    res_out = flag_gems.skip_layer_norm(
        inp, residual, list(layer_shape), weight=weight, bias=bias, eps=eps
    )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skip_rms_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_skip_rmsnorm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape[:2], dtype=dtype, device="cuda")
    residual = torch.randn(shape[:2], dtype=dtype, device="cuda")
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda")
    eps = 1e-5

    ref_inp = to_reference(inp, True)
    ref_residual = to_reference(residual, True)
    ref_weight = to_reference(weight, True)

    def _torch_rms_norm(x, residual, weight, eps):
        x = x + residual
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    ref_out = _torch_rms_norm(
        ref_inp,
        ref_residual,
        weight=ref_weight,
        eps=eps,
    )

    res_out = flag_gems.skip_rms_norm(
        inp, residual, list(layer_shape), weight=weight, eps=eps
    )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.vector_norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize(
    "ord", [2] if QUICK_MODE else [2, float("inf"), -float("inf"), 0, 1]
)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_vectornorm(shape, ord, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.linalg.vector_norm(ref_inp, ord, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.linalg.vector_norm(inp, ord, dim, keepdim)

    gems_assert_close(res_out, ref_out, dtype)
