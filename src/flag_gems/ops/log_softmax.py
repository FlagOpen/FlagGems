import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry

MAX_C_MLU_LOG_SOFTMAX_FORWARD = 16384
MAX_C_MLU_LOG_SOFTMAX_BACKWARD = 32768

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
def log_softmax_kernel(
    output_ptr,
    input_ptr,
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
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = tl.log(numerator / denominator)
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, softmax_output, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD//4}, num_stages=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD//4}, num_stages=5),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD//2}, num_stages=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD//2}, num_stages=5),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD}, num_stages=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD}, num_stages=5),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD//2}, num_stages=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD//2}, num_stages=5),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD}, num_stages=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD}, num_stages=5),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD//4}, num_stages=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD//4}, num_stages=5),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD//2}, num_stages=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD//2}, num_stages=5),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD}, num_stages=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD}, num_stages=5),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD//4}, num_stages=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD//4}, num_stages=5),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD}, num_stages=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_FORWARD}, num_stages=5),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    values={
        "num_warps": lambda args: (
            4 if args["N"] <= 1024 else (8 if args["N"] <= 2048 else 16)
        ),
    },
)
@triton.jit
def log_softmax_kernel_split_c(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    tmp0 = tl.full([BLOCK_M, BLOCK_N], float("-inf"), tl.float32)
    for col_offset in range(0, N, BLOCK_N):
        n_offset = col_offset + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-
                      float("inf")).to(tl.float32)
        # get max for each block
        tmp1 = tl.where(tmp0 < inp, inp, tmp0)
        tmp0 = tmp1

    tmp2 = tl.max(tmp0, 1)[:, None]
    tmp3 = tl.full([BLOCK_M, BLOCK_N], 0, tl.float32)
    for col_offset in range(0, N, BLOCK_N):
        n_offset = col_offset + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        # minus max value each line
        row_minus_max = inp - tmp2
        numerator = tl.exp(row_minus_max)
        tmp4 = tmp3 + numerator
        tmp3 = tmp4

    denominator = tl.sum(tmp3, axis=1)[:, None]
    for col_offset in range(0, N, BLOCK_N):
        n_offset = col_offset + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        row_minus_max = inp - tmp2
        numerator = tl.exp(row_minus_max)
        softmax_output = tl.log(numerator / denominator)
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, softmax_output, mask)


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
def log_softmax_backward_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
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

    offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    out_ptrs = out_ptr + offsets
    out = tl.load(out_ptrs, mask=mask).to(tl.float32)
    out_grad_ptrs = out_grad_ptr + offsets
    out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)

    scale = tl.sum(out_grad, 1)
    in_grad = out_grad - tl.exp(out.to(tl.float32)) * scale[:, None]

    in_grad_ptrs = in_grad_ptr + offsets
    tl.store(in_grad_ptrs, in_grad, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//4}, num_stages=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//4}, num_stages=5),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//4}, num_stages=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//4}, num_stages=5),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//4}, num_stages=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//4}, num_stages=5),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//4}, num_stages=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//4}, num_stages=5),

        triton.Config({"BLOCK_M": 1, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//2}, num_stages=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//2}, num_stages=5),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//2}, num_stages=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//2}, num_stages=5),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//2}, num_stages=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//2}, num_stages=5),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//2}, num_stages=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD//2}, num_stages=5),

        triton.Config({"BLOCK_M": 1, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD}, num_stages=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD}, num_stages=5),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD}, num_stages=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD}, num_stages=5),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD}, num_stages=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD}, num_stages=5),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD}, num_stages=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": MAX_C_MLU_LOG_SOFTMAX_BACKWARD}, num_stages=5),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    values={
        "num_warps": lambda args: (
            4 if args["N"] <= 1024 else (8 if args["N"] <= 2048 else 16)
        ),
    },
)
@triton.jit
def log_softmax_backward_kernel_split_c(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # grad for xn = zn - e^yn * sum(zi) [z for bp grad] and yn = ln(e^xn / sum(e^xi)) for forward
    tmp0 = tl.full([BLOCK_M, BLOCK_N], float(0), tl.float32)
    for col_offset in range(0, N, BLOCK_N):
        n_offset = col_offset + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask,
                           other=float(0)).to(tl.float32)

        tmp1 = tmp0 + out_grad
        tmp0 = tmp1

    scale = tl.sum(tmp0, axis=1)

    for col_offset in range(0, N, BLOCK_N):
        n_offset = col_offset + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N

        out_ptrs = out_ptr + offsets
        out = tl.load(out_ptrs, mask=mask).to(tl.float32)

        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)

        in_grad = out_grad - tl.exp(out.to(tl.float32)) * scale[:, None]

        in_grad_ptrs = in_grad_ptr + offsets
        tl.store(in_grad_ptrs, in_grad, mask=mask)


class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, dtype):
        logging.debug("GEMS LOG_SOFTMAX")

        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        dim = dim % x.ndim
        M = 1
        N = x.shape[dim]
        for i in range(dim):
            M *= x.shape[i]
        inp = x.contiguous()
        if dtype is None:
            dtype = x.dtype
        out = torch.empty_like(inp, dtype=dtype)
        K = inp.numel() // M // N

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        if N > MAX_C_MLU_LOG_SOFTMAX_FORWARD:
            logging.debug(
                "GEMS LOG_SOFTMAX USE SPLITC FORWARD FOR N = %d" % (N))
            log_softmax_kernel_split_c[grid](
                out,
                inp,
                M,
                N,
                K,
            )
        else:
            log_softmax_kernel[grid](
                out,
                inp,
                M,
                N,
                K,
            )
        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS LOG_SOFTMAX VJP")

        dim = ctx.dim
        (out,) = ctx.saved_tensors

        assert dim >= -out.ndim and dim < out.ndim, "Invalid dim"
        dim = dim % out.ndim
        M = 1
        N = out.shape[dim]
        for i in range(dim):
            M *= out.shape[i]

        out_grad = out_grad.contiguous()
        in_grad = torch.empty_like(out)
        K = out.numel() // M // N

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        if N > MAX_C_MLU_LOG_SOFTMAX_BACKWARD:
            logging.debug(
                "GEMS LOG_SOFTMAX USE SPLITC VJP FOR N = %d" % (N))
            log_softmax_backward_kernel_split_c[grid](
                out,
                out_grad,
                in_grad,
                M,
                N,
                K,
            )
        else:
            log_softmax_backward_kernel[grid](
                out,
                out_grad,
                in_grad,
                M,
                N,
                K,
            )
        return in_grad, None, None


def log_softmax(x, dim=-1, dtype=None):
    return LogSoftmax.apply(x, dim, dtype)
