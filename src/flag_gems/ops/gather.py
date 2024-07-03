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


def restride_dim(out, dim, shape):
    strides = list(out.stride())
    strides[dim] = 0
    return out.as_strided(shape, strides)


def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4) for m in block_m
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def gather_kernel(
    inp,
    inp_offsets,
    out,
    out_offsets,
    index,
    idx_offsets,
    M,
    N,
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

        inp_indices = tl.load(inp_offsets + offsets, mask=mask, other=0)
        out_indices = tl.load(out_offsets + offsets, mask=mask, other=0)
        idx_indices = tl.load(idx_offsets + offsets, mask=mask, other=0)

        cur_index = tl.load(index + idx_indices, mask=mask, other=0)
        inp_indices += cur_index * stride_dim
        cur_inp = tl.load(inp + inp_indices, mask=mask, other=0)

        tl.store(out + out_indices, cur_inp, mask=mask)


def gather_out(inp, dim, index, sparse_grad=False, out=None):
    logging.debug("GEMS GATHER OUT")
    assert (
        inp.ndim == index.ndim and inp.ndim == out.ndim
    ), "self, index and out (if it is a Tensor) should all have the same number of dimensions"
    assert (
        (0 <= index.size(i) and index.size(i) <= out.size(i))
        for i in range(0, index.ndim)
    ), "index.size(d) <= out.size(d) for all dimensions d"
    assert (
        ((0 <= index.size(i) and index.size(i) <= inp.size(i)) or i == dim)
        for i in range(0, index.ndim)
    ), "index.size(d) <= self.size(d) for all dimensions d != dim"
    inp = inp.contiguous()
    index = index.contiguous()
    out = out.contiguous()

    out_strided = out.as_strided(index.shape, out.stride())
    inp_strided = restride_dim(inp, dim, index.shape)
    # FIXME: Are there any other way to get the "flatten offset" of a tensor?
    idx = torch.arange(0, index.numel(), device=inp.device).reshape(index.shape)
    # Temporarily call offsetCalculator() outside the block(although it can actually proceed in parallel),
    # because the triton jit.function cannot accept Tuple as input in version 2.2.0(in 3.0.0, it's available),
    # and we do need **the whole stride[]** to accomplish this calculation!
    # FIXME: If stride[] can be wholely passed to triton jit.function, we can do this calculation in the kernel
    # so that the offset calculation can proceed in parallel
    inp_offsets = offsetCalculator(inp_strided, idx, inp.stride(), dim, isInp=True)
    idx_offsets = offsetCalculator(index, idx, index.stride(), dim, isInp=False)
    out_offsets = offsetCalculator(out_strided, idx, out.stride(), dim, isInp=False)
    N = list(index.shape)[index.ndim - 1]
    M = index.numel() // N

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    gather_kernel[grid](
        inp, inp_offsets, out, out_offsets, index, idx_offsets, M, N, inp.stride(dim)
    )
    return out
