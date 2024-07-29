import logging
import math

import torch
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import (
    _get_sort_dim,
    _indicator,
    _is_power_of_two,
    _log2,
    _take_slice,
    zeros_like,
)

from ..utils import libentry


@libentry()
@triton.jit
def topk_stage1_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    k,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_chunk_idx = tl.program_id(1)
    chunk_num = tl.num_programs(1)

    y_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k
    index_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k

    chunk_offset = cur_chunk_idx * CHUNK_SIZE
    x_ptr += cur_batch * N + chunk_offset

    cols = tl.arange(0, CHUNK_SIZE)
    mask = (chunk_offset + cols) < N

    x_val = tl.load(x_ptr + cols, mask=mask, other=-10000.0).to(tl.float32)
    for k_idx in range(k):
        chunk_max_val = tl.max(x_val)
        chunk_max_idx = tl.argmax(x_val, axis=0)
        tl.store(y_ptr + k_idx, chunk_max_val)
        tl.store(index_ptr + k_idx, chunk_max_idx + chunk_offset)
        x_val = tl.where(cols == chunk_max_idx, -10000.0, x_val)


"""
Note(Zhengzekang):
Refer from triton2.2 official `sort` implementation:
https://github.com/triton-lang/triton/blob/release/2.2.x/python/triton/language/standard.py#L392-L404
Just add indices to sort with values.
"""


@triton.jit
def _compare_and_swap(x, ids, desc_mask, n_dims: core.constexpr, idx: core.constexpr):
    l_slice = _take_slice(x, n_dims, idx, 0)
    r_slice = _take_slice(x, n_dims, idx, 1)

    x_int = x
    l_int = l_slice
    r_int = r_slice

    l_idx = _take_slice(ids, n_dims, idx, 0)
    r_idx = _take_slice(ids, n_dims, idx, 1)

    idx_int = ids
    l_int_idx = l_idx
    r_int_idx = r_idx

    # tl.device_print(x)
    if x.dtype.is_floating():
        if core.constexpr(x.dtype.primitive_bitwidth) == 16:
            dtype_int = core.int16
        elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
            dtype_int = core.int32
        elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
            dtype_int = core.int64
        else:
            raise ValueError("Unsupported dtype")
        x_int = x.to(dtype_int, bitcast=True)
        l_int = l_slice.to(dtype_int, bitcast=True)
        r_int = r_slice.to(dtype_int, bitcast=True)

    desc_mask = desc_mask.to(x_int.dtype)
    zero = zeros_like(x_int)
    cond = (l_slice > r_slice) ^ desc_mask
    y = x_int ^ core.where(cond, l_int ^ r_int, zero)
    y = y.to(x.dtype, bitcast=True)

    idx = idx_int ^ core.where(cond, l_int_idx ^ r_int_idx, zeros_like(ids))
    return y, idx


@triton.jit
def _bitonic_merge(
    x,
    ids,
    n_dims: core.constexpr,
    active_dims: core.constexpr,
    order_type: core.constexpr,
):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    core.static_assert(active_dims <= n_dims)

    if order_type == 2:
        desc_mask = _indicator(n_dims, active_dims, 1)
    else:
        desc_mask = order_type

    for i in core.static_range(active_dims):
        x, ids = _compare_and_swap(x, ids, desc_mask, n_dims, active_dims - 1 - i)

    return x, ids


@triton.jit
def argsort(x, ids, dim: core.constexpr = None, descending: core.constexpr = 0):
    core.static_assert(_is_power_of_two(x.shape[_get_sort_dim(dim, x.shape)]))
    core.static_assert(_is_power_of_two(x.numel))
    # reshape the tensor to have all dimensions be 2.
    # TODO: We shouldn't have to change the dimensions not sorted.
    y = core.reshape(x, [2] * _log2(x.numel))
    y_ids = core.reshape(ids, [2] * _log2(ids.numel))

    for i in core.static_range(1, _log2(x.shape[_get_sort_dim(dim, x.shape)]) + 1):
        y, y_ids = _bitonic_merge(
            y,
            y_ids,
            _log2(x.numel),
            i,
            (descending if (i == _log2(x.shape[_get_sort_dim(dim, x.shape)])) else 2),
        )

    x = core.reshape(y, x.shape)
    ids = core.reshape(y_ids, ids.shape)

    return x, ids


@libentry()
@triton.jit
def topk_stage2_kernel(
    y_ptr,
    index_ptr,
    chunk_x,
    chunk_index,
    k: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    chunk_x += cur_batch * N
    chunk_index += cur_batch * N
    y_ptr += cur_batch * k
    index_ptr += cur_batch * k

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    chunk_x_val = tl.load(chunk_x + cols, mask=mask, other=-10000.0).to(tl.float32)
    chunk_index_val = tl.load(chunk_index + cols, mask=mask, other=-10000)

    sorted_chunk_x, sorted_chunk_index = argsort(
        chunk_x_val, chunk_index_val, descending=True
    )
    tl.store(y_ptr + cols, sorted_chunk_x, mask=cols < k)
    tl.store(index_ptr + cols, sorted_chunk_index, mask=cols < k)


def topk(x, k, dim=-1, largest=True, sorted=True):
    logging.debug("GEMS TOPK")
    assert dim == -1, "Currently only support topk in last dimension"
    assert largest, "Currently only support largest == True"
    assert sorted, "Currently only support sorted == True"

    topk_elem_cnt = x.shape[dim]
    batch_size = math.prod(x.shape) // topk_elem_cnt

    chunk_size = 128
    chunk_num = triton.cdiv(topk_elem_cnt, chunk_size)

    stage1_out = torch.empty(batch_size * chunk_num * k, device=x.device, dtype=x.dtype)
    stage1_out_idx = torch.empty(
        batch_size * chunk_num * k, device=x.device, dtype=torch.int32
    )

    out_shape = x.shape[:-1] + (k,)
    stage2_out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    stage2_out_idx = torch.empty(out_shape, device=x.device, dtype=torch.int32)

    topk_stage1_kernel[
        batch_size,
        chunk_num,
    ](
        stage1_out,  # pointer to the output
        stage1_out_idx,  # pointer to the output
        x,  # pointer to the input
        k,
        topk_elem_cnt,
        chunk_size,
    )

    stage2_elem_cnt = chunk_num * k
    BLOCK_SIZE = triton.next_power_of_2(stage2_elem_cnt)

    print(stage2_out.shape, stage2_out.dtype)
    print(stage2_out_idx.shape, stage2_out_idx.dtype)

    print(stage1_out.shape, stage1_out.dtype)
    print(stage1_out_idx.shape, stage1_out_idx.dtype)

    topk_stage2_kernel[batch_size,](
        stage2_out,
        stage2_out_idx,
        stage1_out,
        stage1_out_idx,
        k,
        stage2_elem_cnt,
        BLOCK_SIZE,
    )

    return (stage2_out, stage2_out_idx)
