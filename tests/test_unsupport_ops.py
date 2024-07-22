import logging

import pytest
import torch

import flag_gems

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_SHAPES,
    REDUCTION_SHAPES,
    SCALARS,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)


# ------------------------ test_binary_pointwise_ops.py -------------------------------


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_accuracy_gelu_and_mul(shape, approximate, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(
        torch.nn.functional.gelu(ref_inp1, approximate=approximate), ref_inp2
    )
    with flag_gems.use_gems():
        res_out = flag_gems.gelu_and_mul(inp1, inp2, approximate)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES + [(128, 1024, 1024)])
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES)
@pytest.mark.parametrize("zero_tol", [False, True])
@pytest.mark.parametrize("equal_nan", [False, True])
@pytest.mark.parametrize(
    "gen_nan",
    [0, 1, 2, 3, 4],
)  # 1: nan, 2: inf, 3: -inf, 4: inf vs -inf
def test_accuracy_isclose(shape, dtype, zero_tol, equal_nan, gen_nan):
    rtol = (
        torch.rand(1, dtype=torch.float32, device="musa").item() * 0.0001
        if not zero_tol
        else 0
    )
    if dtype in ALL_FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device="musa")
        inp2 = torch.randn(shape, dtype=dtype, device="musa")
        if gen_nan:
            nan_num = torch.full(
                (1,),
                float("nan" if gen_nan == 1 else "inf"),
                dtype=dtype,
                device="musa",
            )
            inp1.view(-1)[0] = -nan_num if gen_nan == 3 else nan_num
            inp2.view(-1)[0] = -nan_num if gen_nan >= 3 else nan_num
        atol = (
            torch.finfo(dtype).tiny * torch.randint(0, 4, (1,), device="musa").item()
            if not zero_tol
            else 0
        )
    else:
        inp1 = torch.randint(-1000, 1000, shape, device="musa").to(dtype)
        inp2 = torch.randint(-1000, 1000, shape, device="musa").to(dtype)
        if dtype in [torch.int64]:
            inp1.view(-1)[0] = 2**63 - 1
            inp2.view(-1)[0] = -(2**63)
            inp1.view(-1)[1] = 2**60 + 2**20
            inp2.view(-1)[1] = 2**60
            inp1.view(-1)[2] = 2**60 + 1
            inp2.view(-1)[2] = 2**60
            atol = 2 if not zero_tol else 0
            if gen_nan == 0:
                rtol = 0
        elif dtype in [torch.int32]:
            inp1.view(-1)[0] = 2**31 - 1
            inp2.view(-1)[0] = -(2**31)
            inp1.view(-1)[1] = 2**30 + 2**5
            inp2.view(-1)[1] = 2**30
            inp1.view(-1)[2] = 2**30 + 1
            inp2.view(-1)[2] = 2**30
            atol = 2 if not zero_tol else 0
            if gen_nan == 0:
                rtol = 0
        else:
            atol = (
                (
                    torch.finfo(torch.float16).eps
                    * torch.randint(0, 10, (1,), device="musa").item()
                )
                if not zero_tol
                else 0
            )

    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)
    logging.debug(
        "shape={}, dtype={}, rtol={}, atol={}".format(shape, dtype, rtol, atol)
    )

    with flag_gems.use_gems():
        res_out = torch.isclose(inp1, inp2, rtol, atol, equal_nan=equal_nan)
    ref_out = torch.isclose(ref_inp1, ref_inp2, rtol, atol, equal_nan=equal_nan)

    inp1_flat = inp1.view(-1)
    inp2_flat = inp2.view(-1)
    ref_flat = ref_out.view(-1)
    res_flat = res_out.view(-1)
    if dtype in FLOAT_DTYPES and gen_nan:
        logging.debug(
            "equal_nan={}, gen_nan={}: inp1={}, inp2={}, res={}, ref={}".format(
                equal_nan,
                gen_nan,
                inp1_flat[0],
                inp2_flat[0],
                res_flat[0],
                ref_flat[0],
            )
        )
    if dtype in [torch.int64, torch.int32]:
        assert (
            res_flat[1] == ref_flat[1] and res_flat[2] == ref_flat[2]
        ), "res vs ref: {} vs {}, {} vs {}".format(
            res_flat[1], ref_flat[1], res_flat[2], ref_flat[2]
        )
    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES)
@pytest.mark.parametrize("equal_nan", [False, True])
@pytest.mark.parametrize(
    "gen_nan", [0, 1, 2, 3, 4]
)  # 1: nan, 2: inf, 3: -inf, 4: inf vs -inf
def test_accuracy_allclose(shape, dtype, equal_nan, gen_nan):
    rtol = torch.rand(1, dtype=torch.float32, device="musa").item() * (
        0.0001 if dtype in [torch.bfloat16, torch.float16] else 0.01
    )
    if dtype in ALL_FLOAT_DTYPES:
        atol = torch.finfo(dtype).tiny * torch.randint(0, 4, (1,), device="musa").item()
        inp1 = torch.full(shape, 1.234, dtype=dtype, device="musa")
        inp2 = torch.full(shape, 1.234, dtype=dtype, device="musa")
        if gen_nan:
            nan_num = torch.full(
                (1,),
                float("nan" if gen_nan == 1 else "inf"),
                dtype=dtype,
                device="musa",
            )
            inp1.view(-1)[0] = -nan_num if gen_nan == 3 else nan_num
            inp2.view(-1)[0] = -nan_num if gen_nan >= 3 else nan_num
    else:
        atol = (
            torch.finfo(torch.float16).eps
            * torch.randint(0, 10, (1,), device="musa").item()
        )
        inp1 = torch.randint(-1000, 1000, shape, device="musa").to(dtype)
        inp2 = torch.randint(-1000, 1000, shape, device="musa").to(dtype)

    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)
    logging.debug(
        "shape={}, dtype={}, rtol={}, atol={}".format(shape, dtype, rtol, atol)
    )

    with flag_gems.use_gems():
        res_out = torch.allclose(inp1, inp2, rtol, atol, equal_nan=equal_nan)
    ref_out = torch.allclose(ref_inp1, ref_inp2, rtol, atol, equal_nan=equal_nan)

    assert res_out == ref_out


# ------------------------ test_reduction_ops.py -------------------------------


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
        (32, 32, 32, 32, 8), # out of shared-memory
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


# ------------------------ test_special_ops.py -------------------------------


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_randn(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.randn(shape, dtype=dtype, device="musa")
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01
