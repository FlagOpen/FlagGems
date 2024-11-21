import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry, offsetCalculator, restride_dim, TOTAL_CORE_NUM

@libentry()
@triton.autotune(
    configs = [
        triton.Config({"BLOCK_SIZE": 2**n}, num_stages=s)
            for n in range(6, 16, 2)
            for s in [1, 3]
    ],
    key = ["src_elements"],
)
@triton.jit
def select_scatter_2d_kernel(
    inp_ptr,
    src_ptr,
    dim,
    index,
    src_elements,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    job_id = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    start = job_id * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    w_stride = 1 if dim == 0 else W
    h_stride = W if dim == 0 else 1
    for off in range(start, src_elements, step):
        src_offsets = off + tl.arange(0, BLOCK_SIZE)
        src_mask = src_offsets < src_elements
        src = tl.load(src_ptr + src_offsets, mask=src_mask)
        inp_offsets = (off + tl.arange(0, BLOCK_SIZE)) * w_stride + index * h_stride
        tl.store(inp_ptr + inp_offsets, src, mask=src_mask)


def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=1) for m in block_m
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

    if inp.ndim == 2:
        src_elements = src.numel()
        H = out.shape[0]
        W = out.shape[1]
        grid = lambda meta: (min(triton.cdiv(src_elements, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM), )
        select_scatter_2d_kernel[grid](
            out, src, dim, index, src_elements, H, W)
        return out

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
