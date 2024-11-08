import logging
from functools import reduce
from operator import mul

import torch
import triton
import triton.language as tl
from torch import Tensor


def get_output_shape(x: Tensor, offset=0, dim1=-2, dim2=-1):
    x_shape = list(x.shape)
    diag_size = x_shape[-1] + abs(offset)
    rank = x.ndim + 1

    assert dim1 >= -rank and dim1 < rank, f"Invalid dim1: {dim1}"
    assert dim2 >= -rank and dim2 < rank, f"Invalid dim2: {dim2}"
    # convert from negative dims
    dim1 = dim1 % rank
    dim2 = dim2 % rank

    assert dim1 != dim2, "diagonal dimensions cannot be identical"

    # as per the docs, exchanging dims is equivalent to changing the sign of
    # offset
    if dim1 > dim2:
        offset = -offset
        dim1, dim2 = dim2, dim1

    # as per the docs, the size of last dim is placed at dim1 and dim2
    last_dim = x.size(-1) + abs(offset)

    output_shape = [diag_size] * rank
    for dim in range(rank):
        if dim in (dim1, dim2):
            continue
        output_shape[dim] = x_shape.pop(0)
    return output_shape, last_dim, offset


@triton.jit
def diag_embed_kernel(
    x_ptr,
    y_ptr,
    last_dim: tl.constexpr,
    x_cols: tl.constexpr,
    y_cols: tl.constexpr,
    diag_offset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offset1 = tl.arange(0, last_dim)
    mask = offset1 < x_cols
    x_tile = tl.load(x_ptr + offset1 + pid * x_cols, mask=mask, other=0)

    x_tile = x_tile[:, None] if diag_offset > 0 else x_tile[None, :]

    cond = tl.arange(0, last_dim)[:, None] + diag_offset == tl.arange(0, last_dim)
    diag_plane = tl.where(cond, x_tile, 0)

    diag_plane = tl.reshape(diag_plane, (BLOCK_SIZE,))

    output_offset = tl.arange(0, BLOCK_SIZE)
    idx_mask = (tl.arange(0, last_dim)[:, None] < y_cols) & (
        tl.arange(0, last_dim)[None, :] < y_cols
    )
    mask = idx_mask.reshape((BLOCK_SIZE,))
    output_offset = output_offset + pid * y_cols * y_cols
    tl.store(y_ptr + output_offset, diag_plane, mask)


def diag_embed(x: Tensor, offset=0, dim1=-2, dim2=-1) -> Tensor:
    logging.info("GEMS DIAG_EMBED")

    output_shape, last_dim, offset = get_output_shape(x, offset, dim1, dim2)

    y = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    last_dim = triton.next_power_of_2(last_dim)
    grid = lambda meta: (reduce(mul, output_shape[:-2], 1),)
    with torch.cuda.device(x.device):
        diag_embed_kernel[grid](
            x,
            y,
            last_dim,
            x_cols=x.size(-1),
            y_cols=output_shape[-1],
            diag_offset=offset,
            BLOCK_SIZE=last_dim * last_dim,
        )
    return y


a = torch.randn((2, 4), device="cuda")
print(a)
result = diag_embed(a)
print(result)
