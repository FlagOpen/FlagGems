import logging

import torch
import triton
import triton.language as tl
from ..utils import dim_compress

@triton.jit
def scatter_add_kernel_0(
    stride_dim,
    curr_dim_all_stride,
    x_stride,
    out_ptr,
    index_ptr,
    src_ptr,
    N,
    IS_INNER: tl.constexpr,
    xdim_n: tl.constexpr,
    index_dim_n: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * LOOP * BLOCK
    arange = tl.arange(0, BLOCK)
    for loop_iter in tl.static_range(LOOP):
        offsets = block_start + arange
        mask = offsets < N
        cur_index = tl.load(index_ptr + offsets, mask=mask, other=0)
        cur_src = tl.load(src_ptr + offsets, mask=mask, other=0)
        if IS_INNER:
            tmp0 = offsets // index_dim_n * xdim_n
            cur_index = tmp0 + cur_index * stride_dim
        # outer
        else:
            cur_index = (
                cur_index * stride_dim
                + offsets % stride_dim
                + offsets // curr_dim_all_stride * x_stride
            )
        tl.atomic_add(out_ptr + cur_index, cur_src, mask=mask, sem="relaxed")
        block_start += BLOCK


@triton.jit
def scatter_add_kernel_1(
    dim_n,
    out_ptr,
    index_ptr,
    src_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * LOOP
    arange = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + arange
    mask = offsets < n_elements
    for loop_iter in tl.static_range(LOOP):
        src_index_offsets = block_start + arange
        src_tensor = tl.load(src_ptr + src_index_offsets, mask=mask, other=0)
        index_tensor = tl.load(index_ptr + src_index_offsets, mask=mask, other=0)
        out_offsets = src_index_offsets // dim_n * dim_n + index_tensor
        tl.atomic_add(out_ptr + out_offsets, src_tensor, mask=mask, sem="relaxed")
        block_start += BLOCK_SIZE


def get_partial_tensor_with_zeros(tensor, src):
    new_tensor = torch.zeros_like(tensor, dtype=src.dtype)
    slices = []
    target_size = src.size()
    for dim in range(len(tensor.shape)):
        slices.append(slice(0, min(tensor.shape[dim], target_size[dim])))
    new_tensor[tuple(slices)] = src[tuple(slices)]
    return new_tensor


def scatter_add_0(x, dim, index, src):
    logging.debug("GEMS SCATTER ADD")
    dim_n = x.size(dim)
    index_dim_n = index.size(dim)

    all_elem = index.numel()
    grid = lambda meta: (triton.cdiv(all_elem, meta["BLOCK"] * meta["LOOP"]),)
    dim_stride = index.stride(dim)
    cumulative_size = 1

    for i in range(dim, len(index.shape)):
        cumulative_size *= index.shape[i]
    x_stride = 1
    for i in range(dim, len(x.shape)):
        x_stride *= x.shape[i]
    if dim != x.ndim - 1:
        IS_INNER = False
    else:
        IS_INNER = True
    dtype_convert = False
    if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
        dtype_convert = True
        x = x.to(torch.float32)
    scatter_add_kernel_0[grid](
        dim_stride,
        cumulative_size,
        x_stride,
        x,
        index,
        src,
        all_elem,
        IS_INNER=IS_INNER,
        xdim_n=dim_n,
        index_dim_n=index_dim_n,
        BLOCK=128,
        LOOP=4,
    )
    if dtype_convert:
        return x.to(src.dtype)
    return x


def scatter_add_1(x, dim, index, src):
    logging.debug("GEMS SCATTER ADD_")
    dim_n = src.size(dim)
    if dim != x.ndim - 1:
        x = dim_compress(x, dim)
    if dim != x.ndim - 1:
        src = dim_compress(src, dim)
    if dim != x.ndim - 1:
        index = dim_compress(index, dim)

    all_elem = x.numel()
    grid = lambda meta: (triton.cdiv(all_elem, meta["BLOCK_SIZE"] * meta["LOOP"]),)

    dtype_convert = False
    if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
        dtype_convert = True
        x = x.to(torch.float32)
    scatter_add_kernel_1[grid](dim_n, x, index, src, all_elem, BLOCK_SIZE=256, LOOP=1)

    if dim != x.ndim - 1:
        order = [i for i in range(x.ndim - 1)]
        order.insert(dim, x.ndim - 1)
        if dtype_convert:
            return x.permute(order).to(src.dtype)
        return x.permute(order)
    else:
        return x.to(src.dtype)


def scatter_add_(x, dim, index, src):
    logging.debug("GEMS SCATTER ADD_")

    assert x.dim() == index.dim() and x.dim() == src.dim(), "Invalid dim"
    dim = dim % x.ndim
    assert dim >= 0 and dim < x.dim(), "Invalid dim"
    assert index.size(dim) <= src.size(dim), "Invalid src"
    x_count = 0
    index_count = 0
    for d in range(x.dim()):
        if d != dim:
            assert index.size(d) <= x.size(d), "Invalid x"
        if x.size(d) >= src.size(d):
            x_count += 1
        if x.size(d) >= index.size(d):
            index_count += 1

    if x_count == x.dim() and x.shape != src.shape:
        src = get_partial_tensor_with_zeros(x, src)
    if index_count == x.dim() and x.shape != index.shape:
        index = get_partial_tensor_with_zeros(x, index)

    if x.numel() >= 9437184 and dim != x.ndim - 1:
        return scatter_add_1(x, dim, index, src)
    else:
        return scatter_add_0(x, dim, index, src)
