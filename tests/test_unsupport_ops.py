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



# # ------------------------ test_reduction_ops.py -------------------------------


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


@pytest.mark.parametrize("EmbeddingSize", [4096])
@pytest.mark.parametrize("Batch", [2, 4])
@pytest.mark.parametrize("M", [4, 8])
@pytest.mark.parametrize("N", [128, 256, 4096])
@pytest.mark.parametrize("padding_idx", [None, -1, 1, 2])
@pytest.mark.parametrize("scale_grad_by_freq", [True, False])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32]
)  # triton.atomic_add still not support bf16
def test_embedding(EmbeddingSize, Batch, M, N, padding_idx, scale_grad_by_freq, dtype):
    indices = torch.randint(
        0, EmbeddingSize, (Batch, M), device="musa", requires_grad=False
    )
    embedding = torch.randn(
        (EmbeddingSize, N), device="musa", dtype=dtype, requires_grad=True
    )
    ref_indices = to_reference(indices)
    ref_embedding = to_reference(embedding)

    ref_out = torch.nn.functional.embedding(
        ref_indices, ref_embedding, padding_idx, scale_grad_by_freq=scale_grad_by_freq
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.embedding(
            indices, embedding, padding_idx, scale_grad_by_freq=scale_grad_by_freq
        )
    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_embedding, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, embedding, out_grad)

    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)