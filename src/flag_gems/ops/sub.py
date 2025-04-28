import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 8}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 16}, num_warps=4),
    ],
    key=["n_elements"],
)
@triton.jit
def sub_func_tensor_scalar_block(
    input_ptr, b: tl.constexpr, output_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    input_val = tl.load(input_ptr + offsets, mask=mask)

    tl.store(output_ptr + offsets, input_val - b * alpha, mask=mask)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def sub_func(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_tensor_scalar(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_scalar_tensor(x, y, alpha):
    return x - y * alpha


def sub(A, B, *, alpha=1):
    logging.debug("GEMS SUB")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return sub_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return sub_func_tensor_scalar(A, B, alpha)
        # input_ = A.contiguous()
        # n_elements = input_.numel()
        # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        # output = torch.empty(A.shape, device=A.device, dtype=A.dtype)
        # sub_func_tensor_scalar_block[grid](input_, B, output, alpha, n_elements)
        # return output
    elif isinstance(B, torch.Tensor):
        return sub_func_scalar_tensor(A, B, alpha)
    else:
        # Both scalar
        return torch.tensor(A - B * alpha)


def sub_(A, B, *, alpha=1):
    logging.debug("GEMS SUB_")
    if isinstance(B, torch.Tensor):
        return sub_func(A, B, alpha, out0=A)
    else:
        return sub_func_tensor_scalar(A, B, alpha, out0=A)
