import logging

import torch
import triton
import copy
import triton.language as tl
import triton.backends.mlu.driver as driver
import math

from ..utils import libentry, TOTAL_CLUSTER_NUM, TOTAL_CORE_NUM, MAX_NRAM_SIZE

MAX_C_MLU_SOFTMAX_FORWARD = 16384
# The maximum size value for an m×n when pipelining is enabled, this is a heuristic value.
MAX_MXN_SOFTMAX_FORWARD_FOR_INNER = 18560
MAX_MXN_SOFTMAX_FORWARD_FOR_NON_INNER = 20480
MAX_C_MLU_SOFTMAX_BACKWARD = 8192
MLU_GRID_MAX = 65535

def max_multiple_less_than(a, b):
    if a < b:
        return a
    return (a // b) * b

def config_prune1(configs, named_args, **kwargs):
    M = named_args["M"]
    K = named_args["K"]
    N = named_args["N"]
    configs_map = {}
    pruned_configs = []
    # When N is less than MAX_C_MLU_SOFTMAX_FORWARD, no reduction loops
    doopt = N < MAX_MXN_SOFTMAX_FORWARD_FOR_NON_INNER
    for config in configs:
        kw = config.kwargs
        TILE_K, TILE_N, num_warps, num_stages = \
            kw['TILE_K'], kw['TILE_N'], config.num_warps, config.num_stages
        if doopt:
            config = copy.deepcopy(config)
            TILE_N = config.kwargs["TILE_N"] = N
            k_per_core = math.ceil(K / max(TOTAL_CORE_NUM // M, 1))
            # The usage of nram is three times the size of the input/output while pipline is not enabled.
            nram_usage = k_per_core * N * 4 * 3
            if nram_usage < MAX_NRAM_SIZE:
                TILE_K = config.kwargs["TILE_K"] = k_per_core
                num_stages = config.num_stages = 1
            else:
                max_block_k = MAX_MXN_SOFTMAX_FORWARD_FOR_NON_INNER // N
                align_num = 64 // 4
                max_block_k = max_multiple_less_than(max_block_k, align_num)
                TILE_K = config.kwargs["TILE_K"] = max_block_k
                num_stages = config.num_stages = 3
        
        key = (TILE_K, TILE_N, num_warps, num_stages)
        # Only keep one config for the same key
        configs_map.setdefault(key, config)
    pruned_configs = []
    for k, v in configs_map.items():
        pruned_configs.append(v)
    added_config = copy.deepcopy(pruned_configs[0])
    added_config.kwargs["TILE_K"] = 1
    added_config.kwargs["TILE_N"] = N
    added_config.num_warps = 1
    added_config.num_stages = 3
    pruned_configs.append(added_config)
    return pruned_configs

def softmax_tile_mode1(args):
    one_tile_k = args["TILE_K"] * max(TOTAL_CORE_NUM // args["M"], 1) >= args["K"]
    one_tile_n = args["TILE_N"] >= args["N"]
    if one_tile_n and one_tile_k:
        return 0
    elif one_tile_n and not one_tile_k:
        return 1
    else:
        return 2

@triton.autotune(
    configs=[
        triton.Config({
            "TILE_K": k,
            "TILE_N": 2**n
        },
                      num_stages=s,
                      num_warps=1) for k in [1, 2, 4, 8]
        for n in range(10, 15, 1) for s in [1, 3]
    ],
    key=[
        "N",
        "K",
    ],
    prune_configs_by={'early_config_prune': config_prune1},
)
@triton.heuristics(
    values={
        "TILE_MODE": lambda args: softmax_tile_mode1(args),
    }, )
@triton.jit
def softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_MODE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    p_k_num = tl.num_programs(axis=1)
    split_k = tl.cdiv(K, p_k_num)
    k_start = pid_k * split_k

    if TILE_MODE == 0:
        n_offset = tl.arange(0, TILE_N)
        k_offset = pid_k * TILE_K + tl.arange(0, TILE_K)
        offset = pid_m * N * K + n_offset[:, None] * K + k_offset[None, :]
        mask = (n_offset[:, None] < N) & (k_offset[None, :] < K)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask,
                      other=-float("inf")).to(tl.float32)
        row_minus_max = inp - tl.max(inp, axis=0)[None, :]
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)[None, :]
        recip = 1.0 / denominator
        softmax_output = numerator * recip
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, softmax_output, mask=mask)
    elif TILE_MODE == 1:
        for k_idx in range(0, split_k, TILE_K):
            k_offset = k_start + k_idx + tl.arange(0, TILE_K)            
            n_offset = tl.arange(0, TILE_N)
            offset = pid_m * N * K + n_offset[:, None] * K + k_offset[None, :]
            mask = k_offset[None, :] < K and n_offset[:, None] < N
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask,
                          other=-float("inf")).to(tl.float32)
            row_minus_max = inp - tl.max(inp, axis=0)[None, :]
            numerator = tl.exp(row_minus_max)
            denominator = tl.sum(numerator, axis=0)[None, :]
            recip = 1.0 / denominator
            softmax_output = numerator * recip
            output_ptrs = output_ptr + offset
            tl.store(output_ptrs, softmax_output, mask=mask)
    else:
        for k_idx in range(0, split_k, TILE_K):
            k_offset = k_start + k_idx + tl.arange(0, TILE_K)
            m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
            z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

            # specialization does not improve performance inn this example, as tested
            for start_n in range(0, N, TILE_N):
                n_offset = start_n + tl.arange(0, TILE_N)
                offset = pid_m * N * K + n_offset[:, None] * K + k_offset[None, :]
                mask = (n_offset[:, None] < N) & (k_offset[None, :] < K)
                inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
                m_new = tl.maximum(m, inp)
                alpha = tl.exp(m - m_new)
                z = z * alpha + tl.exp(inp - m_new)
                m = m_new

            m_reduced = tl.max(m, 0)  # (TILE_K,)
            z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)  # (TILE_K, )
            recip_z = 1.0 / z
            m = m_reduced

            # specialization does not improve performance inn this example, as tested
            previous_multiple = prev_multiple_of(N, TILE_N)
            for start_n in range(0, N, TILE_N):
                n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
                offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
                mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
                inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
                o = tl.exp(inp - m[None, :]) * recip_z[None, :]
                tl.store(output_ptr + offsets, o, mask=mask)


def config_prune2(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    configs_map = {}
    pruned_configs = []
    # When N is less than MAX_C_MLU_SOFTMAX_FORWARD, no reduction loops
    doopt = N < MAX_MXN_SOFTMAX_FORWARD_FOR_INNER
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, num_warps, num_stages = \
            kw['BLOCK_M'], kw['BLOCK_N'], config.num_warps, config.num_stages
        if doopt:
            config = copy.deepcopy(config)
            BLOCK_N = config.kwargs["BLOCK_N"] = N
            m_per_core = math.ceil(M / TOTAL_CORE_NUM)
            # The usage of nram is three times the size of the input/output while pipline is not enabled.
            nram_usage = m_per_core * N * 4 * 3
            if nram_usage < MAX_NRAM_SIZE:
                BLOCK_M = config.kwargs["BLOCK_M"] = m_per_core
                num_stages = config.num_stages = 1
            else:
                max_block_m = MAX_MXN_SOFTMAX_FORWARD_FOR_INNER // N
                align_num = 64 // 4
                max_block_m = max_multiple_less_than(max_block_m, align_num)
                BLOCK_M = config.kwargs["BLOCK_M"] = max_block_m
                num_stages = config.num_stages = 3
        key = (BLOCK_M, BLOCK_N, num_warps, num_stages)
        # Only keep one config for the same key
        configs_map.setdefault(key, config)
    pruned_configs = []
    for k, v in configs_map.items():
        pruned_configs.append(v)
    # Add a heuristic config.
    added_config = copy.deepcopy(pruned_configs[0])
    added_config.kwargs["BLOCK_M"] = 1
    added_config.kwargs["BLOCK_N"] = N
    added_config.num_warps = 1
    added_config.num_stages = 3
    pruned_configs.append(added_config)
    return pruned_configs

def softmax_tile_mode2(args):
    one_tile_m = args["BLOCK_M"] * TOTAL_CORE_NUM >= args["M"]
    one_tile_n = args["BLOCK_N"] >= args["N"]
    if one_tile_n and one_tile_m:
        return 0
    elif one_tile_n and not one_tile_m:
        return 1
    else:
        return 2

@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": m,
            "BLOCK_N": 2**n
        },
                      num_stages=s,
                      num_warps=1) for m in [1, 2, 4, 8]
        for n in range(10, 15, 1) for s in [1, 3]
    ],
    key=[
        "M",
        "N",
    ],
    prune_configs_by={'early_config_prune': config_prune2},
)
@triton.heuristics(
    values={
        "TILE_MODE": lambda args: softmax_tile_mode2(args),
    }, )
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_MODE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pnum = tl.num_programs(axis=0)
    split_m = tl.cdiv(M, pnum)
    m_start = pid_m * split_m

    if TILE_MODE == 0:
        m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offset = tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask,
                      other=-float("inf")).to(tl.float32)
        row_minus_max = inp - tl.max(inp, axis=1)[:, None]
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=1)[:, None]
        recip = 1.0 / denominator
        softmax_output = numerator * recip
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, softmax_output, mask=mask)
    elif TILE_MODE == 1:
        for m_idx in range(0, split_m, BLOCK_M):
            m_offset = m_start + m_idx + tl.arange(0, BLOCK_M)
            n_offset = tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N + n_offset[None, :]
            mask = m_offset[:, None] < M and n_offset[None, :] < N
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask,
                          other=-float("inf")).to(tl.float32)
            trans_inp = tl.trans(inp)
            row_minus_max = trans_inp - tl.max(trans_inp, axis=0)[None, :]
            numerator = tl.exp(row_minus_max)
            denominator = tl.sum(numerator, axis=0)[None, :]
            recip = 1.0 / denominator
            softmax_output = tl.trans(numerator * recip)
            output_ptrs = output_ptr + offset
            tl.store(output_ptrs, softmax_output, mask=mask)
    else:
        for m_idx in range(0, split_m, BLOCK_M):
            m_offset = m_start + m_idx + tl.arange(0, BLOCK_M)
            block_max = tl.full([BLOCK_M, BLOCK_N], value=float("-inf"), dtype=tl.float32)
            block_sum = tl.full([BLOCK_M, BLOCK_N], value=0.0, dtype=tl.float32)
            # specialization does not improve performance inn this example, as tested
            for start_n in range(0, N, BLOCK_N):
                n_offset = start_n + tl.arange(0, BLOCK_N)
                offset = m_offset[:, None] * N + n_offset[None, :]
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp = tl.load(input_ptr + offset, mask=mask,
                              other=-float("inf")).to(tl.float32)
                cur_max = tl.maximum(block_max, inp)
                alpha = tl.exp(block_max - cur_max)
                block_sum = block_sum * alpha + tl.exp(inp - cur_max)
                block_max = cur_max
            
            trans_block_max = tl.trans(block_max)
            trans_block_sum = tl.trans(block_sum)
            max_reduced = tl.max(trans_block_max, 0)
            total_sum = tl.sum(trans_block_sum * tl.exp(trans_block_max - max_reduced[None, :]), 0)
            recip_total_sum = 1.0 / total_sum
            total_max = max_reduced

            for start_n in range(0, N, BLOCK_N):
                n_offset = start_n + tl.arange(0, BLOCK_N)
                offset = m_offset[:, None] * N + n_offset[None, :]
                mask = m_offset[:, None] < M and n_offset[None, :] < N
                inp = tl.load(input_ptr + offset, mask=mask,
                              other=-float("inf")).to(tl.float32)
                o = tl.exp(inp - total_max[:, None]) * recip_total_sum[:, None]
                tl.store(output_ptr + offset, o, mask=mask)



# ------------------------  backward -------------------------------
@libentry()
@triton.autotune(
    configs=[
        triton.Config({"TILE_K": 2**k}, num_warps=1, num_stages = s)
        for k in range(3, 11, 1)
        for s in [1, 3]
    ],
    key=[
        "M",
        "N",
        "K",
    ],
)
@triton.heuristics(
    values={
        "TILE_K":
        lambda args: max(triton.cdiv(args["K"], MLU_GRID_MAX), args["TILE_K"])
    })
@triton.heuristics(
    values={
        "TILE_N": lambda args: max(1, MAX_C_MLU_SOFTMAX_BACKWARD // args["TILE_K"]),
        "ONE_TILE_PER_CTA": lambda args: args["TILE_N"] >= args["N"],
    },
)
@triton.jit
def softmax_backward_kernel_non_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offsets_k = pid_k * TILE_K + tl.arange(0, TILE_K)

    # grad for xn = zn * yn - yn * sum(yi * zi) [z for bp grad] and yn = e^xn / sum(e^xi) for forward
    if ONE_TILE_PER_CTA:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        mask = (offsets_n < N)[:, None] & (offsets_k < K)
        out_tile = tl.load(out_ptr + offsets, mask=mask)
        out_tile = tl.cast(out_tile, tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
        out_grad_tile = tl.cast(out_grad_tile, tl.float32)
        scale = tl.sum(out_tile * out_grad_tile, axis=0)
        in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        scale = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            scale += out_tile * out_grad_tile
            offsets_n += TILE_N
            offsets += TILE_N * K
        scale = tl.sum(scale, axis=0)  # (TILE_K)

        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            offsets_n += TILE_N
            offsets += TILE_N * K


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"TILE_N": MAX_C_MLU_SOFTMAX_BACKWARD // (2**k)}, num_warps=1, num_stages = s)
        for k in range(0, 6, 1)
        for s in [1, 3]
    ],
    key=["M", "N"],
)
@triton.heuristics(
    values={
        "TILE_M": lambda args: max(1, MAX_C_MLU_SOFTMAX_BACKWARD // args["TILE_N"]),
        "ONE_TILE_PER_CTA": lambda args: args["TILE_N"] >= args["N"],
    },
)
@triton.jit
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        out_tile = tl.load(out_ptr + offsets, mask=mask)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
        scale = tl.sum(out_tile * out_grad_tile, 1)
        in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_last"
            )
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            scale += out_tile * out_grad_tile
            n_offsets += TILE_N
            offsets += TILE_N
        scale = tl.sum(scale, 1)  # (TILE_M,)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_first"
            )
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            n_offsets += TILE_N
            offsets += TILE_N

class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, dtype):
        logging.debug("GEMS SOFTMAX")

        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        dim = dim % x.ndim
        M = 1
        N = x.shape[dim]
        for i in range(dim):
            M *= x.shape[i]  # pre_dim
        inp = x.contiguous()
        if dtype is None:
            dtype = x.dtype
        out = torch.empty_like(inp, dtype=dtype)
        K = inp.numel() // M // N  # post_dim

        with torch.mlu.device(inp.device):
            if K > 1:
                logging.debug("GEMS SOFTMAX USE NON INNER")
                grid = lambda meta: (M, max(TOTAL_CORE_NUM // M, 1), 1)
                softmax_kernel_non_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                    K,
                )
            else:
                logging.debug("GEMS SOFTMAX USE INNER")
                softmax_kernel_inner[TOTAL_CORE_NUM, 1, 1](
                    out,
                    inp,
                    M,
                    N,
                )
        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS SOFTMAX VJP")
        dim = ctx.dim
        (out,) = ctx.saved_tensors

        assert dim >= -out.ndim and dim < out.ndim, "Invalid dim"
        dim = dim % out.ndim
        M = 1
        N = out.shape[dim]
        for i in range(dim):
            M *= out.shape[i]

        out_grad = out_grad.contiguous()
        in_grad = torch.empty_like(out)
        K = out.numel() // M // N

        with torch.mlu.device(in_grad.device):
            if K > 1:
                logging.debug("GEMS SOFTMAX VJP USE NON INNER")
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                softmax_backward_kernel_non_inner[grid](
                    out,
                    out_grad,
                    in_grad,
                    M,
                    N,
                    K,
                )
            else:
                logging.debug("GEMS SOFTMAX VJP USE INNER")
                grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
                softmax_backward_kernel_inner[grid](
                    out,
                    out_grad,
                    in_grad,
                    M,
                    N,
                )
        return in_grad, None, None


def softmax(x, dim=-1, dtype=None):
    return Softmax.apply(x, dim, dtype)
