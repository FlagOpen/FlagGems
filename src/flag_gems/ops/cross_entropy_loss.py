import torch
import triton
import triton.language as tl
import logging
from ..utils import libentry
from .sum import sum


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}, num_stages=4),
        triton.Config({"BLOCK_M": 1}, num_stages=5),
        triton.Config({"BLOCK_M": 2}, num_stages=4),
        triton.Config({"BLOCK_M": 2}, num_stages=5),
        triton.Config({"BLOCK_M": 4}, num_stages=4),
        triton.Config({"BLOCK_M": 4}, num_stages=5),
        triton.Config({"BLOCK_M": 8}, num_stages=4),
        triton.Config({"BLOCK_M": 8}, num_stages=5),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    values={
        "BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
        "num_warps": lambda args: (
            4 if args["N"] <= 1024 else (8 if args["N"] <= 2048 else 16)
        ),
    },
)
@triton.jit
def log_softmax_and_mul_kernel(
    output_ptr,
    input_ptr,
    target_ptr,
    all_mean,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    input_ptrs = input_ptr + offset
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = tl.log(numerator / denominator)
    target = tl.load(target_ptr + offset, mask=mask, other=0.0)
    out = softmax_output * target / (all_mean) * -1
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}, num_stages=4),
        triton.Config({"BLOCK_M": 1}, num_stages=5),
        triton.Config({"BLOCK_M": 2}, num_stages=4),
        triton.Config({"BLOCK_M": 2}, num_stages=5),
        triton.Config({"BLOCK_M": 4}, num_stages=4),
        triton.Config({"BLOCK_M": 4}, num_stages=5),
        triton.Config({"BLOCK_M": 8}, num_stages=4),
        triton.Config({"BLOCK_M": 8}, num_stages=5),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    values={
        "BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
        "num_warps": lambda args: (
            4 if args["N"] <= 1024 else (8 if args["N"] <= 2048 else 16)
        ),
    },
)
@triton.jit
def softmax_and_sub_kernel(
    output_ptr,
    input_ptr,
    target_ptr,
    out_grad,
    all_mean,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    input_ptrs = input_ptr + offset
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    # todo: reduce unnecessary calculations through mask operations to improve performance
    softmax_output = numerator / denominator
    target_ptrs = target_ptr + offset
    target = tl.load(target_ptrs, mask=mask, other=0.0)
    out_grad_value = tl.load(out_grad)
    out = out_grad_value * (softmax_output - target) / all_mean
    output_ptrs = output_ptr + offset

    tl.store(output_ptrs, out, mask=mask)


class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, weight, reduction, ignore_index, label_smoothing):
        logging.debug("GEMS CrossEntropyLoss")
        assert isinstance(input, torch.Tensor), "input is not a tensor"
        if input.ndim >= 2:
            dim = 1
        else:
            dim = 0

        shape = list(input.shape)
        shape[dim] = 1
        all_mean = target.numel()
        target = torch.zeros_like(input).scatter(dim, target.view(shape), 1)

        M = 1
        N = input.shape[dim]
        for i in range(dim):
            M *= input.shape[i]
        inp = input.contiguous()
        out = torch.empty_like(inp, dtype=inp.dtype)
        K = inp.numel() // M // N

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        log_softmax_and_mul_kernel[grid](
            out,
            inp,
            target,
            all_mean,
            M,
            N,
            K,
        )
        out_result = sum(out)

        ctx.save_for_backward(input, target)
        ctx.dim = dim
        ctx.mean = all_mean
        return out_result

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS CrossEntropyLoss VJP")
        input, target = ctx.saved_tensors
        dim = ctx.dim
        all_mean = ctx.mean

        M = 1
        N = input.shape[dim]
        for i in range(dim):
            M *= input.shape[i]
        inp = input.contiguous()
        out = torch.empty_like(inp, dtype=input.dtype)
        K = inp.numel() // M // N

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        softmax_and_sub_kernel[grid](
            out,
            inp,
            target,
            out_grad,
            all_mean,
            M,
            N,
            K,
        )
        return out, None, None, None, None, None


# todo: reducetion(dtype: int,default mean->1), support other scenarios as follows: (none->0, sum->2)
def cross_entropy_loss(
    input, target, weight=None, reduction="Mean", ignore_index=-100, label_smoothing=0.0
):
    return CrossEntropyLoss.apply(
        input, target, weight, reduction, ignore_index, label_smoothing
    )
