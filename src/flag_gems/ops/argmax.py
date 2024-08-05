import logging
import math

import torch
import triton
import triton.language as tl

from ..utils import libentry, cfggen_reduce_op, TOTAL_CORE_NUM
from ..utils.shape_utils import can_use_int32_index


@libentry()
@triton.autotune(configs=cfggen_reduce_op(), key=["M"])
@triton.jit
def argmax_kernel_1(
    inp,
    out,
    M,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    _tmp = tl.full([BLOCK_SIZE], value=-float("inf"), dtype=tl.float32)
    block_start = block_start.to(tl.int64)
    for off in range(block_start, M, step):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp + offset, mask=mask, other=-float("inf"))
        _tmp = tl.where((_tmp < inp_val), inp_val, _tmp)

    max_val = tl.max(_tmp, axis=0, return_indices=False)
    tl.atomic_max(out, max_val)


def heur_block_n(args):
    return min(4096, triton.next_power_of_2(args["N"]))


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 8}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=5),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def argmax_kernel(
    inp,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    # set offset
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if INT64_INDEX:
        pid_m = pid_m.to(tl.int64)
        pid_k = pid_k.to(tl.int64)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    max_values = tl.full([BLOCK_M], dtype=tl.float32, value=float("-inf"))
    argmax_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
        local_max, local_argmax = tl.max(
            inp_vals, 1, return_indices=True, return_indices_tie_break_left=True
        )
        # if return indices is not supported, call a tl.argmax in addition
        # local_argmax = tl.argmax(inp_vals, 1)
        update = local_max > max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(update, start_n + local_argmax, argmax_values)

    offset_index = m_offset * K + pid_k
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_index_ptrs, argmax_values, mask=mask1)


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logging.debug("GEMS ARGMAX")
    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype
        use_int64_index = not can_use_int32_index(inp)

        grid = lambda meta: (min(triton.cdiv(M, meta['BLOCK_SIZE']), TOTAL_CORE_NUM),)
        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.full(shape, float("-inf"), dtype=torch.float32, device=inp.device)
        else:
            out = torch.full([], float("-inf"), dtype=torch.float32, device=inp.device)

        with torch.mlu.device(inp.device):
            argmax_kernel_1[grid](inp, out, M, INT64_INDEX=use_int64_index)
        return out.to(torch.int64)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        shape = inp.shape
        dim = dim % inp.ndim
        N = shape[dim]
        M = math.prod(shape[:dim])
        K = inp.numel() // M // N

        inp = inp.contiguous()
        use_int64_index = not can_use_int32_index(inp)

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        with torch.mlu.device(inp.device):
            argmax_kernel[grid](inp, out_index, M, N, K, INT64_INDEX=use_int64_index)

        return out_index
