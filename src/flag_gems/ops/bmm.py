import logging

import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

from ..utils import libentry


def heur_divisible_m(args):
    return args["M"] % args["BLOCK_M"] == 0


def heur_divisible_n(args):
    return args["N"] % args["BLOCK_N"] == 0


def heur_divisible_k(args):
    return args["K"] % args["BLOCK_K"] == 0


def get_configs_io_bound():
    configs = []
    for num_stages in [1, 2]:
        for block_m in (
            [16, 32]
            if torch.version.hip is None and not hasattr(torch, "corex")
            else [32, 64]
        ):
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    for group_m in [1, 2]:
                        num_warps = 4 if block_n <= 64 else 8
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_K": block_k,
                                    "GROUP_M": group_m,
                                    "SPLIT_K": 1,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )
    return configs


def get_configs_compute_bound():
    configs = []
    for num_stages in [1, 2]:
        for block_m in [64, 128, 256]:
            for block_n in [64, 128, 256]:
                for block_k in [32, 64, 128]:
                    for group_m in [1, 2]:
                        num_warps = 8 if block_n <= 64 else 16
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_K": block_k,
                                    "GROUP_M": group_m,
                                    "SPLIT_K": 1,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )
    return configs


def get_nv_configs():
    configs = []
    if hasattr(torch, "corex"):
        return configs
    configs = [
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 1, "SPLIT_K": 1},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 2, "SPLIT_K": 1},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 2, "SPLIT_K": 1},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 2, "SPLIT_K": 1},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 2, "SPLIT_K": 1},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 2, "SPLIT_K": 1},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 1, "SPLIT_K": 1},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 2, "SPLIT_K": 1},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 2, "SPLIT_K": 1},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 2, "SPLIT_K": 1},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 2, "SPLIT_K": 1},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 2, "SPLIT_K": 1},
            num_warps=4,
            num_stages=3,
        ),
    ]
    return configs


@libentry()
@triton.autotune(
    # get_nv_configs() + get_configs_compute_bound() + get_configs_io_bound(),
    # get_configs_compute_bound(),
    configs=get_configs_io_bound(),
    key=["M", "N", "K"],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 10,
    },
)
@triton.heuristics(
    {
        "DIVISIBLE_M": heur_divisible_m,
        "DIVISIBLE_N": heur_divisible_n,
        "DIVISIBLE_K": heur_divisible_k,
    }
)
@triton.jit
def bmm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    # batch offsets
    pid_b = tl.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    C += pid_b * M * N

    pidx = tl.program_id(0)
    pidy = tl.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = tl.num_programs(0)
        gridy = tl.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M

        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        if (group_id * GROUP_M + GROUP_M) > gridx:
            GROUP_SIZE = gridx % GROUP_M
        else:
            GROUP_SIZE = GROUP_M
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = C + offs_m[:, None] * N + offs_n[None, :]

    num_iters = tl.cdiv(K, BLOCK_K)
    o = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(num_iters):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask_a)
        b = tl.load(b_ptrs, mask_b)

        offs_k += BLOCK_K
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

        o += tl.dot(a, b, allow_tf32=False)

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M and not DIVISIBLE_N:
        mask_c = mask_n[None, :]
    elif not DIVISIBLE_M and DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, o, mask_c)


def bmm(A, B):
    logging.debug("GEMS BMM")
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["BLOCK_M"]),
        triton.cdiv(meta["N"], meta["BLOCK_N"]),
        batch,
    )
    with torch.cuda.device(A.device):
        bmm_kernel[grid_fn](A, B, out, M, N, K)
    return out
