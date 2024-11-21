import logging
import math

import torch
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2, zeros_like

from ..utils import libentry, MAX_GRID_SIZE_X, TOTAL_CORE_NUM

_MIN_FLOAT32_VAL: tl.constexpr = torch.finfo(torch.float32).min
_MAX_FLOAT32_VAL: tl.constexpr = torch.finfo(torch.float32).max
_MIN_FLOAT16_VAL: tl.constexpr = torch.finfo(torch.float16).min
_MAX_FLOAT16_VAL: tl.constexpr = torch.finfo(torch.float16).max
_MIN_BFLOAT16_VAL: tl.constexpr = torch.finfo(torch.bfloat16).min
_MAX_BFLOAT16_VAL: tl.constexpr = torch.finfo(torch.bfloat16).max
_MIN_INT32_VAL: tl.constexpr = torch.iinfo(torch.int32).min
_MAX_INT32_VAL: tl.constexpr = torch.iinfo(torch.int32).max


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
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

    # tl.device_print("shape is: ", shape)
    y = core.reshape(x, shape)
    y_idx = core.reshape(ids, shape)

    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(x.dtype)
    right = core.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(x.dtype)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    left_idx = core.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape).to(
        ids.dtype
    )
    right_idx = core.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape).to(
        ids.dtype
    )
    left_idx = core.reshape(left_idx, ids.shape)
    right_idx = core.reshape(right_idx, ids.shape)

    # actual compare-and-swap
    if core.constexpr(x.dtype.primitive_bitwidth) == 8:
        idtype = core.int8
    elif core.constexpr(x.dtype.primitive_bitwidth) == 16:
        idtype = core.int16
    elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
        idtype = core.int32
    elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
        idtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip
    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    if core.constexpr(ids.dtype.primitive_bitwidth) == 8:
        idx_dtype = core.int8
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 16:
        idx_dtype = core.int16
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 32:
        idx_dtype = core.int32
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 64:
        idx_dtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft_idx = left_idx.to(idx_dtype, bitcast=True)
    iright_idx = right_idx.to(idx_dtype, bitcast=True)
    ix_idx = ids.to(idx_dtype, bitcast=True)
    ret_idx = ix_idx ^ core.where(cond, ileft_idx ^ iright_idx, zeros_like(ix_idx))

    return ret.to(x.dtype, bitcast=True), ret_idx.to(ids.dtype, bitcast=True)


@triton.jit
def _bitonic_merge(
    x, ids, stage: core.constexpr, order: core.constexpr, n_dims: core.constexpr
):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: core.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x, ids, dim: tl.constexpr, descending: core.constexpr):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = dim
    n_dims: core.constexpr = _log2(x.shape[_dim])
    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


@libentry()
@triton.jit
def topk_stage2_kernel(
    y_ptr,
    index_ptr,
    chunk_x,
    chunk_index,
    sort_dim: tl.constexpr,
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
    chunk_index_val = tl.load(chunk_index + cols, mask=mask, other=mask_index_val).to(
        tl.int32
    )

    sorted_chunk_x, sorted_chunk_index = argsort(
        chunk_x_val, chunk_index_val, 0, descending=DESCENDING
    )
    tl.store(y_ptr + cols, sorted_chunk_x, mask=cols < k)
    tl.store(index_ptr + cols, sorted_chunk_index, mask=cols < k)


@triton.jit
def get_topk_bubble_res(buffer, buffer_ind, k, axis, mask_val, DESCENDING,
                     BLOCK_M, BLOCK_N):
    kep_buffer_n = buffer
    topk_buffer_index_n = buffer_ind
    ret = tl.empty([BLOCK_M, k], dtype=buffer.dtype)
    ret_ind = tl.empty([BLOCK_M, k], dtype=buffer_ind.dtype)
    for k_ind in tl.range(0, k):
        if DESCENDING:
            sel_val, sel_index = tl.max(kep_buffer_n,
                                        axis=axis,
                                        return_indices=True)
        else:
            sel_val, sel_index = tl.min(kep_buffer_n,
                                        axis=axis,
                                        return_indices=True)

        if BLOCK_M > 1:
            mask_sel = tl.arange(0, BLOCK_N)[None, :] == sel_index[:, None]
            tep_sel_index_buffer = tl.where(mask_sel, topk_buffer_index_n, 0)
            sel_index_res = tl.max(tep_sel_index_buffer, axis=axis)
            sel_val_res = sel_val
            ret[:, k_ind] = sel_val_res
            ret_ind[:, k_ind] = sel_index_res

            # Update buffer.
            kep_buffer_n = tl.where(mask_sel, mask_val, kep_buffer_n)
        else:
            indices = sel_index[0]
            ret[:, k_ind] = sel_val
            ret_ind[:, k_ind] = topk_buffer_index_n[:, indices]
            # Update buffer.
            kep_buffer_n[:, indices] = mask_val
    return ret, ret_ind


BLOCK_BATCH = [1, 4, 16, 64]
BLOCK_N = [128, 256, 512, 1024, 2048]


def topk_cfggen():
    num_stage = [1, 3]
    configs = [
        triton.Config({
            "TILE_M": m,
            "TILE_N": n
        }, num_warps=1, num_stages=s) for m in BLOCK_BATCH for n in BLOCK_N
        for s in num_stage
    ]
    return configs


def topk_config_prune(configs, named_args, **kwargs):
    k = named_args["k"]
    N = named_args["N"]
    block_m = named_args["BLOCK_M"]
    new_configs = []

    for config in configs:
        tile_n = config.kwargs["TILE_N"]
        tile_m = config.kwargs["TILE_M"]
        if tile_n < k or tile_m > block_m:
            continue
        if len(new_configs) >= 1:
            last_tn = new_configs[-1].kwargs["TILE_N"]
            last_tm = new_configs[-1].kwargs["TILE_M"]
            if tile_n > N and last_tn >= N and last_tm == tile_m:
                continue
        config.kwargs["TILE_M_NUM"] = triton.cdiv(block_m, tile_m)
        config.kwargs["TILE_N_NUM"] = triton.cdiv(N, tile_n)
        new_configs.append(config)

    if not N in BLOCK_N and N <= max(BLOCK_N):
        for tm in BLOCK_BATCH:
            new_configs.append(
                triton.Config(
                    {
                        "TILE_M": tm,
                        "TILE_N": N,
                        "TILE_M_NUM": triton.cdiv(block_m, tm),
                        "TILE_N_NUM": 1
                    },
                    num_warps=1,
                    num_stages=3))
    return new_configs


@libentry()
@triton.autotune(configs=topk_cfggen(),
                 key=["k", "N", "M", "BLOCK_M", "DESCENDING"],
                 prune_configs_by={"early_config_prune": topk_config_prune})
@triton.jit
def topk_bubble_kernel(
    inp_ptr,
    out_ptr,
    out_index_ptr,
    k: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_M_NUM: tl.constexpr,
    TILE_N_NUM: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    pid = tl.program_id(0)
    m_st = pid * BLOCK_M

    mask_val = _get_finfo_val(inp_ptr.dtype.element_ty,
                              return_max=not DESCENDING)
    mask_val = mask_val.to(inp_ptr.dtype.element_ty)

    for m_block_ind in tl.range(0, TILE_M_NUM):
        m_iter_st = m_block_ind * TILE_M + m_st
        m_offset_val = m_iter_st + tl.arange(0, TILE_M)
        m_offset = m_offset_val[:, None]
        m_offset_mask = m_offset < M

        topk_buffer_n = tl.full([TILE_M, TILE_N_NUM * k],
                                value=mask_val,
                                dtype=inp_ptr.dtype.element_ty)
        topk_buffer_index_n = tl.full([TILE_M, TILE_N_NUM * k],
                                      value=0,
                                      dtype=out_index_ptr.dtype.element_ty)
        for n_block_ind in tl.range(0, TILE_N_NUM):
            n_st = n_block_ind * TILE_N
            n_offset = n_st + tl.arange(0, TILE_N)[None, :]
            n_offset_mask = n_offset < N

            inp_mask = m_offset_mask & n_offset_mask
            inp_ptrs = inp_ptr + m_offset * N + n_offset
            block_inp_val = tl.load(inp_ptrs, mask=inp_mask, other=mask_val)

            local_buffer, local_buffer_ind = get_topk_bubble_res(
                block_inp_val, n_offset.to(out_index_ptr.dtype.element_ty), k,
                1, mask_val, DESCENDING, TILE_M, TILE_N)
            tep_index = n_block_ind * k
            topk_buffer_n[:, tep_index:tep_index + k] = local_buffer
            topk_buffer_index_n[:, tep_index:tep_index + k] = local_buffer_ind

        global_res, global_res_ind = get_topk_bubble_res(
            topk_buffer_n, topk_buffer_index_n, k, 1, mask_val, DESCENDING,
            TILE_M, TILE_N_NUM * k)

        # Store topk.
        store_ptrs = m_offset * k + tl.arange(0, k)[None, :]
        store_mask = m_offset_mask
        tl.store(store_ptrs + out_ptr, global_res, store_mask)
        tl.store(store_ptrs + out_index_ptr, global_res_ind, store_mask)



def topk(x, k, dim=-1, largest=True, sorted=True):
    logging.debug("GEMS TOPK")
    # If dim equals to last dim, we set it to -1.
    if dim < 0:
        dim = dim + x.ndim

    assert dim == x.ndim - 1, "Currently only support topk in last dimension"
    assert sorted, "Currently only support sorted == True"

    descending = True
    if not largest:
        descending = False

    topk_elem_cnt = x.shape[dim]
    batch_size = math.prod(x.shape) // topk_elem_cnt
    out_shape = x.shape[:-1] + (k,)

    if k <= math.log2(topk_elem_cnt):
        logging.debug("GEMS TOPK USING BUBBLE")
        topk_out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        topk_out_idx = torch.empty(out_shape,
                                   device=x.device,
                                   dtype=torch.int64)

        def grid_fn(meta):
            return (min(batch_size, TOTAL_CORE_NUM), )

        block_m = triton.cdiv(batch_size, TOTAL_CORE_NUM)
        topk_bubble_kernel[grid_fn](x,
                                    topk_out,
                                    topk_out_idx,
                                    k,
                                    batch_size,
                                    topk_elem_cnt,
                                    block_m,
                                    DESCENDING=descending)
        return (topk_out, topk_out_idx)
    else:
        logging.debug("GEMS TOPK USING SORT")
        # Note(Zhengzekang): Maybe we should add a heuristic search in selecting a proper chunk size.
        if topk_elem_cnt < 1024:
            chunk_size = 256
        else:
            chunk_size = 1024

        # Note(Zhengzekang): We should promise chunk_size is larger than k.
        if chunk_size < k:
            chunk_size = triton.next_power_of_2(k)

        chunk_num = triton.cdiv(topk_elem_cnt, chunk_size)

        stage1_out = torch.empty(batch_size * chunk_num * k, device=x.device, dtype=x.dtype)
        stage1_out_idx = torch.empty(
            batch_size * chunk_num * k, device=x.device, dtype=torch.int64
        )

        stage2_out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        stage2_out_idx = torch.empty(out_shape, device=x.device, dtype=torch.int64)

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
                dim,
                k,
                stage2_elem_cnt,
                BLOCK_SIZE,
                descending,
            )

        return (stage2_out, stage2_out_idx)
