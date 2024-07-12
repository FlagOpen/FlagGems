import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry

MAX_C_MLU_CUMSUM = 16384


def config_prune(configs, named_args, **kwargs):
    N = named_args['N']
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, num_warps, num_stages = \
            kw['BLOCK_M'], kw['BLOCK_N'], config.num_warps, config.num_stages
        # When N is less than MAX_C_MLU_CUMSUM, no reduction loops
        if N < MAX_C_MLU_CUMSUM:
            BLOCK_N = kw['BLOCK_N'] = N
            num_stages = config.num_stages = 1
        key = (BLOCK_M, BLOCK_N, num_warps, num_stages)
        # Only keep one config for the same key
        configs_map.setdefault(key, config)
    pruned_configs = []
    for k, v in configs_map.items():
        pruned_configs.append(v)
    configs = pruned_configs
    return pruned_configs

@libentry()
@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_M": m,
            "BLOCK_N": 2**n
        },
                      num_stages=s,
                      num_warps=1) for m in range(1, 30, 3)
        for n in range(6, 13, 1) for s in [1, 3]
    ],
    key=[
        "M",
        "N",
        "K",
    ],
    prune_configs_by={'early_config_prune': config_prune},
)
@triton.jit
def cumsum_kernel_split(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    kep = tl.full([BLOCK_M, BLOCK_N], float(0), tl.float32)
    for col_offset in range(0, N, BLOCK_N):
        n_offset = col_offset + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * \
            K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=float(0)).to(tl.float32)

        raw_res = tl.cumsum(inp_vals, axis=1)
        kep_tmp = kep[:, BLOCK_N - 1]
        result = raw_res + kep_tmp[:, None]
        kep = result

        out_ptrs = out + offset
        tl.store(out_ptrs, result, mask=mask)


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
    out = torch.empty_like(inp, dtype=dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch.mlu.device(inp.device):
        cumsum_kernel_split[grid](inp, out, M, N, K)
    return out
