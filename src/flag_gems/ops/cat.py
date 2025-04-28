# import itertools
import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl


@triton.jit
def cat_kernel(
    in_ptr,
    out_ptr,
    in_strides_ptr,
    out_strides_ptr,
    shape_ptr,
    offset,
    total_elements,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    linear_id = offs
    in_offset = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    out_offset = tl.full([BLOCK_SIZE], offset, dtype=tl.int32)

    for d in range(ndim):
        shape_d = tl.load(shape_ptr + d)
        in_stride = tl.load(in_strides_ptr + d)
        out_stride = tl.load(out_strides_ptr + d)

        coord = linear_id % shape_d
        linear_id = linear_id // shape_d

        in_offset += coord * in_stride
        out_offset += coord * out_stride

    val = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + out_offset, val, mask=mask)


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logging.debug("TRITON CAT")

    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return A[0]

    dim = dim % A[0].ndim
    base_shape = A[0].shape

    for t in A:
        if t.ndim != len(base_shape):
            raise RuntimeError("Tensors must have the same number of dimensions")
        for i in range(t.ndim):
            if i != dim and t.shape[i] != base_shape[i]:
                raise RuntimeError(f"Size mismatch at dim {i}")

    if all(t.numel() == 0 for t in A):
        empty_shape = list(base_shape)
        empty_shape[dim] = 0
        return torch.empty(empty_shape, dtype=A[0].dtype, device=A[0].device)

    out_shape = list(base_shape)
    out_shape[dim] = sum(t.shape[dim] for t in A)
    out = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)

    out_strides = torch.tensor(out.stride(), dtype=torch.int32, device=out.device)
    offset = 0

    for t in A:
        if t.numel() == 0:
            continue

        in_strides = torch.tensor(t.stride(), dtype=torch.int32, device=t.device)
        shape_tensor = torch.tensor(t.shape, dtype=torch.int32, device=t.device)
        total_elements = t.numel()

        grid = lambda META: (triton.cdiv(total_elements, META["BLOCK_SIZE"]),)
        cat_kernel[grid](
            in_ptr=t,
            out_ptr=out,
            in_strides_ptr=in_strides,
            out_strides_ptr=out_strides,
            shape_ptr=shape_tensor,
            offset=offset,
            total_elements=total_elements,
            ndim=t.ndim,
            BLOCK_SIZE=8,
        )

        offset += t.shape[dim] * out.stride()[dim]

    return out
