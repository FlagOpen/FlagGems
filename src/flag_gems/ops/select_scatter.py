import logging

import torch

from ..ops.copy import copy
from ..utils.shape_utils import has_internal_overlapping
=======
import triton
import triton.language as tl

from ..utils import libentry, offsetCalculator, restride_dim


def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4) for m in block_m
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def select_scatter_kernel(
    inp,
    inp_indices,
    src,
    src_offsets,
    M,
    N,
    index,
    stride_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M

    for off in range(0, N, BLOCK_N):
        cols_offsets = off + tl.arange(0, BLOCK_N)[None, :]
        cols_mask = cols_offsets < N

        offsets = rows_offsets * N + cols_offsets
        mask = rows_mask and cols_mask

        indices = tl.load(inp_indices + offsets, mask=mask, other=0)
        src_indices = tl.load(src_offsets + offsets, mask=mask, other=0)
        cur_src = tl.load(src + src_indices, mask=mask, other=0)

        indices += index * stride_dim
        tl.store(inp + indices, cur_src, mask=mask)
>>>>>>> 89c65c7 ([Operator] slice&select scatter (#143))


def select_scatter(inp, src, dim, index):
    logging.debug("GEMS SELECT_SCATTER")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index >= -inp.size(dim) and index < inp.size(dim), "Invalid index"
    dim = dim % inp.ndim
    index = index % inp.size(dim)

    valid_shape = list(inp.shape)
    del valid_shape[dim]
    assert (
        list(src.shape) == valid_shape
    ), "Expected src to have a size equal to the slice of self"

    if has_internal_overlapping(inp):
        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
    else:
        out = torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )

    copy(inp, out0=out)
    indices = [slice(None)] * inp.ndim
    indices[dim] = index
    copy(src, out0=out[indices])

    return out
