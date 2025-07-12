import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def index_select_kernel(
    inp_ptr,
    out_ptr,
    index_ptr,
    dim_prod_post,
    dim_size_in,
    dim_size_out,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < total_elements

    idx = block_start + offsets
    pre_idx = idx // (dim_size_out * dim_prod_post)
    out_dim_idx = (idx // dim_prod_post) % dim_size_out
    post_idx = idx % dim_prod_post

    in_dim_idx = tl.load(index_ptr + out_dim_idx)

    inp_offset = (
        pre_idx * dim_size_in * dim_prod_post + in_dim_idx * dim_prod_post + post_idx
    )

    inp = tl.load(inp_ptr + inp_offset, mask=mask)
    tl.store(out_ptr + idx, inp, mask=mask)


def index_select(inp, dim, index):
    logger.debug("GEMS INDEX SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    # The check below is a generator, it needs to be consumed.
    # Using all() to consume it and check all values.
    assert all((i >= 0 and i < inp.size(dim)) for i in index), "Index out of range"

    if index.ndim == 0:
        index = index.unsqueeze(0)

    dim = dim % inp.ndim
    inp_shape = inp.shape
    index_len = index.numel()

    out_shape = list(inp_shape)
    out_shape[dim] = index_len
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    if out.numel() == 0:
        return out

    dim_prod_post = 1
    for s in inp_shape[dim + 1 :]:
        dim_prod_post *= s

    total_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    index_select_kernel[grid](
        inp,
        out,
        index,
        dim_prod_post,
        inp_shape[dim],
        index_len,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
