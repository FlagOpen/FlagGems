import logging

import torch
import triton
import copy
import triton.language as tl

from ..utils import libentry, TOTAL_CORE_NUM

#FIXME(cambricon): double 8192 when JIRA:1488 is fixed
MAX_C_MLU_CUMSUM = 8192

def config_prune(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, TILE_N, num_warps, num_stages = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            kw["TILE_N"],
            config.num_warps,
            config.num_stages,
        )
        new_config = config
        # When N is less than MAX_C_MLU_CUMSUM, no reduction loops. Unify different BLOCK_N configs.
        if N <= MAX_C_MLU_CUMSUM:
            # change config
            new_config = copy.deepcopy(config)
            BLOCK_N = new_config.kwargs["BLOCK_N"] = triton.next_power_of_2(N)
            num_stages = new_config.num_stages = 1
        else:
            # When N is greater than MAX_C_MLU_CUMSUM, the pruning condition was obtained through experimentation.
            # It may result in not finding the optimal solution.
            if BLOCK_N < 2048:
                continue
            if BLOCK_N >= 2048 and TILE_N < 8:
                continue
            if BLOCK_N < MAX_C_MLU_CUMSUM and BLOCK_M < M and BLOCK_M <= (
                    MAX_C_MLU_CUMSUM // BLOCK_N * 2):
                continue
        # BLOCK_M can only be 1 when BLOCK_N is at its maximum
        if BLOCK_N == MAX_C_MLU_CUMSUM and BLOCK_M > 1:
            continue
        # Prune invalid BLOCK_M
        if BLOCK_M > M:
            continue
        # Prune invalid TILE_N
        if TILE_N > BLOCK_N:
            continue
        # The pruning condition was obtained through experimentation. It may result in not finding the optimal solution.
        if BLOCK_N > 128 and TILE_N < 8:
            continue
        key = (BLOCK_M, BLOCK_N, TILE_N, num_warps, num_stages)
        # Only keep one config for the same key
        configs_map.setdefault(key, new_config)
    pruned_configs = []
    for k, v in configs_map.items():
        pruned_configs.append(v)
    return pruned_configs

@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": m,
            "BLOCK_N": 2**n,
            "TILE_N": 2**t,
        },
                      num_stages=s,
                      num_warps=1) for m in range(1, 30, 3)
        for n in range(7, 14, 1) for t in range(0, 7, 1) for s in [1, 3]
    ],
    key=[
        "M",
        "N",
        "K",
    ],
    prune_configs_by={'early_config_prune': config_prune},
)
@triton.heuristics(
    values={
        "TILE_NUM":
        lambda args: args["BLOCK_N"] // args["TILE_N"] if args["BLOCK_N"] %
        args["TILE_N"] == 0 and args["BLOCK_N"] // args["TILE_N"] >= 1 else 1,
        "TILE_N":
        lambda args: args["BLOCK_N"]
        if args["TILE_NUM"] == 1 else args["TILE_N"],
    }, )
@triton.jit
def cumsum_blelloch(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_NUM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # FIXME(cambricon): shape [BLOCK_M] is enough, here shape [BLOCK_M, BLOCK_N] is a
    # workaround for compiler bug
    kep = tl.full([BLOCK_M, BLOCK_N], float(0), tl.float32)
    for col_offset in range(0, N, BLOCK_N):
        n_offset = col_offset + tl.arange(0, BLOCK_N)
        # Pointers to the start of the row
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        x_ptrs = inp + offsets
        y_ptrs = out + offsets

        # Load data into NRAM
        in_block = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        x_block = tl.reshape(in_block, (BLOCK_M, TILE_NUM, TILE_N))
        # Trans TILE_N and apply blelloch in TILE_N dim
        x_block = tl.trans(x_block, 0, 2, 1)

        # Apply blelloch algo
        # Up-Sweep Phase
        step = 1
        while step < TILE_N:
            idx_a = step - 1
            idx_b = idx_a + step
            while idx_b < TILE_N:
                x_block[:,
                        idx_b, :] = x_block[:, idx_a, :] + x_block[:, idx_b, :]
                idx_a += 2 * step
                idx_b += 2 * step
            step *= 2

        # Down-Sweep Phase
        step //= 2
        while step > 0:
            idx_b = TILE_N - 1 - step
            idx_a = idx_b - step
            while idx_a > 0:
                x_block[:,
                        idx_b, :] = x_block[:, idx_a, :] + x_block[:, idx_b, :]
                idx_b -= 2 * step
                idx_a -= 2 * step
            step //= 2

        # Deal the last tile row exclusive sum(Composed by right shift and tl.cumsum)
        # Right shift 1 position for the last tile row
        partial_sum = tl.zeros((BLOCK_M, TILE_NUM), dtype=tl.float32)
        partial_sum[:, 1:] = x_block[:, TILE_N - 1, 0:(TILE_NUM - 1)]
        partial_sum = tl.cumsum(partial_sum, axis=1)

        # Apply cycle add for all tile data
        x_block += partial_sum[:, None, :]

        # Trans TILE_N dim to original pos
        x_block = tl.trans(x_block, 0, 2, 1)
        x_block = tl.reshape(x_block, (BLOCK_M, BLOCK_N))
        # Add last block partial sum to current block
        kep_tmp = kep[:, BLOCK_N - 1]
        x_block = x_block + kep_tmp[:, None]
        # Store current result to kep for next block calculation
        kep = x_block
        # Store result back to global memory
        tl.store(y_ptrs, x_block, mask=mask)

def cumsum(inp, dim=1, *, dtype=None):
    logging.debug("GEMS CUMSUM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    inp = inp.contiguous()
    K = inp.numel() // M // N

    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64
    out = torch.empty_like(inp, dtype=dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch.mlu.device(inp.device):
        cumsum_blelloch[grid](inp, out, M, N, K)
    return out
