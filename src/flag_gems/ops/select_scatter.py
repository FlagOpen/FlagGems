import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry


def offsetCalculator(inp, idx, strides, dim, isInp):
    ndim = inp.ndim
    shape = list(inp.shape)
    offsets = 0
    idx_dim = 0
    for d in range(0, ndim):
        mod = torch.floor(idx % shape[d])
        add_on = mod * strides[d]
        offsets += add_on
        if d == dim:
            idx_dim = add_on
        idx = idx // shape[d]
        # FIXME: Should we write a fast div/mod
        # to boost the '%' and '//'? (Since they may be run many times)
        # See also:
        #   - https://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html
        #   - Division by Invariant Integers Using Multiplication,
        #     Torbjörn Granlund and Peter L. Montgomery, 1994.
    return (offsets) if not isInp else (offsets - idx_dim)


def restride_dim(src, dim, shape, step=0):
    strides = list(src.stride())
    strides[dim] *= step
    return src.as_strided(shape, strides)


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


def select_scatter(inp, src, dim, index):
    logging.debug("GEMS SELECT_SCATTER")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index >= -inp.size(dim) and index < inp.size(dim), "Invalid index"
    dim = dim % inp.ndim
    index = index % inp.size(dim)
    out = inp.clone().contiguous()
    src = src.contiguous()

    valid_shape = list(inp.shape)
    del valid_shape[dim]
    assert (
        list(src.shape) == valid_shape
    ), "Expected src to have a size equal to the slice of self"

    src_expanded_shape = list(inp.shape)
    src_expanded_shape[dim] = 1
    out_strided = restride_dim(out, dim, src_expanded_shape)
    idx = torch.arange(0, src.numel(), device=inp.device).reshape(src_expanded_shape)
    indices = offsetCalculator(
        out_strided, idx, out.stride(), dim, isInp=False
    ).squeeze(dim=dim)
    src_offsets = offsetCalculator(src, idx, src.stride(), dim, isInp=False).squeeze(
        dim=dim
    )

    N = valid_shape[src.ndim - 1]
    M = src.numel() // N

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    select_scatter_kernel[grid](
        out, indices, src, src_offsets, M, N, index, out.stride(dim)
    )

    return out
