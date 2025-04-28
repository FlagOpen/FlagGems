import logging

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [4, 8, 16, 32]],
    key=["n_elements"],
)
@triton.jit
def silu_forward_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid

    tl.store(output_ptr + offsets, y, mask=mask)


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [4, 8, 16, 32]],
    key=["n_elements"],
)
@triton.jit
def silu_backward_kernel(
    grad_ptr, input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    dy = tl.load(grad_ptr + offsets, mask=mask)

    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    dx = dy * sigmoid * (1.0 + x * (1.0 - sigmoid))

    tl.store(output_ptr + offsets, dx, mask=mask)


class Silu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("Triton Silu Forward")
        out = torch.empty_like(A)
        n_elements = A.numel()
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        silu_forward_kernel[grid](out, A, n_elements)
        ctx.save_for_backward(A)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("Triton Silu Backward")
        (inp,) = ctx.saved_tensors
        in_grad = torch.empty_like(inp)
        n_elements = inp.numel()
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        silu_backward_kernel[grid](out_grad, inp, in_grad, n_elements)
        return in_grad


def silu(A: torch.Tensor) -> torch.Tensor:
    return Silu.apply(A)


class InplaceSilu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logging.debug("Triton Inplace Silu Forward")
        ctx.save_for_backward(A.clone())  # backup for backward
        ctx.mark_dirty(A)
        n_elements = A.numel()
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        silu_forward_kernel[grid](A, A, n_elements)
        return A

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("Triton Inplace Silu Backward")
        (inp,) = ctx.saved_tensors
        in_grad = torch.empty_like(inp)
        n_elements = inp.numel()
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        silu_backward_kernel[grid](out_grad, inp, in_grad, n_elements)
        return in_grad


def silu_(A: torch.Tensor) -> torch.Tensor:
    InplaceSilu.apply(A)
    return A
