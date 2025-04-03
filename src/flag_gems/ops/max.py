import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from .. import runtime
from ..runtime import torch_device_fn
from ..utils import libentry
from ..utils import triton_lang_extension as tle


# @libentry()
@triton.jit
def max_kernel_1(
    inp, # 输入
    mid, # 每块的规约输出
    M, # 元素数
    BLOCK_SIZE: tl.constexpr, # 块大小
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M # 掩码
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf")) # 加载数据
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)


# @libentry()
@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


# @libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("max"),
    key=[
        "M",
        "N",
    ],
)
@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    result_value = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    result_index = tl.zeros([BLOCK_M], dtype=tl.int64)
    for i in range(0, N, BLOCK_N):
        n_offset = i + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        # set mask
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
        max_value, max_index = tl.max(inp_vals, axis=1, return_indices=True)
        update_mask = max_value > result_value
        result_value = tl.where(update_mask, max_value, result_value)
        result_index = tl.where(update_mask, i + max_index, result_index)
    mask1 = m_offset < M
    offset_index = m_offset * K + pid_k
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)


def max(inp):
    logging.debug("GEMS MAX")
    inp = inp.contiguous()
    M = inp.numel() # 456192
    block_size_1 = min(64, triton.next_power_of_2(math.ceil(math.sqrt(M)))) # 64
    mid_size_1 = triton.cdiv(M, block_size_1) # 7128
    #block_mid_1 = triton.next_power_of_2(mid_size_1) # 8192

    dtype = inp.dtype
    mid_1 = torch.empty((mid_size_1,), dtype=dtype, device=inp.device)
    
    # import pdb; pdb.set_trace()
    # with torch_device_fn.device(inp.device):
    max_kernel_1[(mid_size_1, 1, 1)](inp, mid_1, M, block_size_1) # 分组规约，每组大小block_size，共block_mid组，每组的最大值保存到mid_1里面
    
    block_size_2 = triton.next_power_of_2(math.ceil(math.sqrt(mid_size_1))) # 128
    mid_size_2 = triton.cdiv(mid_size_1, block_size_2) # 56
    block_mid_2 = triton.next_power_of_2(mid_size_2) # 64
    
    mid_2 = torch.empty((mid_size_2,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    max_kernel_1[(mid_size_2, 1, 1)](mid_1, mid_2, mid_size_1, block_size_2)

    max_kernel_2[(1, 1, 1)](mid_2, out, mid_size_2, block_mid_2) # 再从mid_2中规约一次，求出最大值
    return out


def max_dim(inp, dim=None, keepdim=False):
    logging.debug("GEMS MAX DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    # with torch_device_fn.device(inp.device):
    max_kernel[grid](inp, out_value, out_index, M, N, K)
    Max_out = namedtuple("max", ["values", "indices"])
    out = Max_out(values=out_value, indices=out_index)
    return out
