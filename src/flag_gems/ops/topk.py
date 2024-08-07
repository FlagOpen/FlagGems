import logging
import math

import torch
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import (
    _is_power_of_two,
    _log2,
    _unwrap_if_constexpr,
    zeros_like,
)

from ..utils import libentry

_MIN_FLOAT32_VAL = torch.finfo(torch.float32).min
_MAX_FLOAT32_VAL = torch.finfo(torch.float32).max
_MIN_FLOAT16_VAL = torch.finfo(torch.float16).min
_MAX_FLOAT16_VAL = torch.finfo(torch.float16).max
_MIN_BFLOAT16_VAL = torch.finfo(torch.bfloat16).min
_MAX_BFLOAT16_VAL = torch.finfo(torch.bfloat16).max
_MIN_INT32_VAL = torch.iinfo(torch.int32).min
_MAX_INT32_VAL = torch.iinfo(torch.int32).max


def _get_sort_dim(dim, shape):
    dim = _unwrap_if_constexpr(dim)
    shape = _unwrap_if_constexpr(shape)
    if dim is None:
        dim = len(shape) - 1
    assert dim == len(shape) - 1, "Currently only support sorting on the last dimension"
    return core.constexpr(dim)


@triton.jit
def _indicator(n_dims: core.constexpr, idx: core.constexpr, pos: core.constexpr):
    core.static_assert(idx < n_dims)
    core.static_assert((pos == 0) or (pos == 1))
    y = core.arange(0, 2)
    if pos == 0:
        y = 1 - y

    for n in core.static_range(0, n_dims):
        if n != n_dims - 1 - idx:
            y = core.expand_dims(y, n)
    return y


@triton.jit
def _take_slice(
    x,
    n_dims: core.constexpr,
    idx: core.constexpr,
    pos: core.constexpr,
    keep_dim: core.constexpr = True,
):
    y = triton.language.standard.sum(x * _indicator(n_dims, idx, pos), n_dims - 1 - idx)
    if keep_dim:
        y = core.expand_dims(y, n_dims - 1 - idx)

    return y


@triton.jit
def _get_finfo_val(
    dtype,
    return_max,
):
    if dtype is tl.float32:
        if return_max:
            return _MAX_FLOAT32_VAL
        else:
            return _MIN_FLOAT32_VAL
    elif dtype is tl.float16:
        if return_max:
            return _MAX_FLOAT16_VAL
        else:
            return _MIN_FLOAT16_VAL
    elif dtype is tl.bfloat16:
        if return_max:
            return _MAX_BFLOAT16_VAL
        else:
            return _MIN_BFLOAT16_VAL


@libentry()
@triton.jit
def topk_stage1_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    k,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
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

    mask_val = _get_finfo_val(x_ptr.dtype.element_ty, return_max=not DESCENDING)
    x_val = tl.load(x_ptr + cols, mask=mask, other=mask_val).to(tl.float32)
    for k_idx in range(k):
        if DESCENDING:
            chunk_select_val = tl.max(x_val)
            chunk_select_idx = tl.argmax(x_val, axis=0)
        else:
            chunk_select_val = tl.min(x_val)
            chunk_select_idx = tl.argmin(x_val, axis=0)

        tl.store(y_ptr + k_idx, chunk_select_val)
        tl.store(index_ptr + k_idx, chunk_select_idx + chunk_offset)

        if DESCENDING:
            x_val = tl.where(
                cols == chunk_select_idx,
                _get_finfo_val(tl.float32, return_max=False),
                x_val,
            )
        else:
            x_val = tl.where(
                cols == chunk_select_idx,
                _get_finfo_val(tl.float32, return_max=True),
                x_val,
            )


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
    DESCENDING: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    chunk_x += cur_batch * N
    chunk_index += cur_batch * N
    y_ptr += cur_batch * k
    index_ptr += cur_batch * k

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    mask_val = _get_finfo_val(chunk_x.dtype.element_ty, return_max=not DESCENDING)
    mask_index_val = _MIN_INT32_VAL if DESCENDING else _MAX_INT32_VAL

    chunk_x_val = tl.load(chunk_x + cols, mask=mask, other=mask_val).to(tl.float32)
    chunk_index_val = tl.load(chunk_index + cols, mask=mask, other=mask_index_val)

    sorted_chunk_x, sorted_chunk_index = argsort(
        chunk_x_val, chunk_index_val, descending=DESCENDING
    )
    tl.store(y_ptr + cols, sorted_chunk_x, mask=cols < k)
    tl.store(index_ptr + cols, sorted_chunk_index, mask=cols < k)


def topk(x, k, dim=-1, largest=True, sorted=True):
    logging.debug("GEMS TOPK")
    # If dim equals to last dim, we set it to -1.
    if dim == x.ndim - 1:
        dim = -1

    assert dim == -1, "Currently only support topk in last dimension"
    assert sorted, "Currently only support sorted == True"

    descending = True
    if not largest:
        descending = False

    topk_elem_cnt = x.shape[dim]
    batch_size = math.prod(x.shape) // topk_elem_cnt

    # Note(Zhengzekang): Maybe we should add a heuristic search in selecting a proper chunk size.
    if topk_elem_cnt < 1024:
        chunk_size = 256
    else:
        chunk_size = 1024

    chunk_num = triton.cdiv(topk_elem_cnt, chunk_size)

    stage1_out = torch.empty(batch_size * chunk_num * k, device=x.device, dtype=x.dtype)
    stage1_out_idx = torch.empty(
        batch_size * chunk_num * k, device=x.device, dtype=torch.int32
    )

    out_shape = x.shape[:-1] + (k,)
    stage2_out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    stage2_out_idx = torch.empty(out_shape, device=x.device, dtype=torch.int32)

    with torch.cuda.device(x.device):
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
            descending,
        )

    stage2_elem_cnt = chunk_num * k
    BLOCK_SIZE = triton.next_power_of_2(stage2_elem_cnt)

    with torch.cuda.device(x.device):
        topk_stage2_kernel[batch_size,](
            stage2_out,
            stage2_out_idx,
            stage1_out,
            stage1_out_idx,
            k,
            stage2_elem_cnt,
            BLOCK_SIZE,
            descending,
        )

    return (stage2_out, stage2_out_idx)
