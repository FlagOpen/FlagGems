import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def copy_func_kernel(
    out_ptr,
    in_ptr,
    dim_size_in,
    dim_size_out,
    dim_prod_post,
    dim_offset,
    total_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = tl.arange(0, BLOCK)
    mask = block_start + offsets < total_elements

    idx = block_start + offsets

    pre_idx = idx // (dim_size_in * dim_prod_post)
    dim_idx = (idx // dim_prod_post) % dim_size_in
    post_idx = idx % dim_prod_post

    out_idx = (
        pre_idx * dim_size_out * dim_prod_post
        + (dim_idx + dim_offset) * dim_prod_post
        + post_idx
    )

    data = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS CAT")
    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return A[0]

    assert dim >= -A[0].ndim and dim < A[0].ndim, f"Invalid dim: {dim}"
    # Convert negative dim to positive
    dim = dim % A[0].ndim

    # Same rank check
    inp_shapes = [list(_.shape) for _ in A]
    inp0_shape = inp_shapes[0]
    for s in inp_shapes[1:]:
        if len(s) != len(inp0_shape):
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(s)}"
            )
    # Same size check
    for tensor_idx, inp_shape in enumerate(inp_shapes):
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx == dim:
                continue
            elif length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    dtype = A[0].dtype
    device = A[0].device
    ndims = A[0].ndim
    shapes = [t.shape for t in A]

    cat_dim_sizes = [s[dim] for s in shapes]
    out_shape = list(shapes[0])
    out_shape[dim] = sum(cat_dim_sizes)
    out = torch.empty(out_shape, dtype=dtype, device=device)

    BLOCK = 1024
    dim_offset = 0

    for i, tensor in enumerate(A):
        tensor_shape = tensor.shape

        dim_size_in = tensor_shape[dim]
        dim_size_out = out_shape[dim]

        dim_prod_post = 1
        for d in range(dim + 1, ndims):
            dim_prod_post *= tensor_shape[d]

        total_elements = tensor.numel()

        grid = lambda meta: (triton.cdiv(total_elements, BLOCK),)

        copy_func_kernel[grid](
            out,
            tensor,
            dim_size_in,
            dim_size_out,
            dim_prod_post,
            dim_offset,
            total_elements,
            BLOCK=BLOCK,
        )

        dim_offset += dim_size_in

    return out
