import torch
import triton
import triton.language as tl
import logging
from ..utils import libentry


@libentry()
@triton.autotune(
    configs=[
        triton.Config(
            {"TILE_M": 32, "TILE_N": 32, "TILE_K": 32, "GROUP_M": 1},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"TILE_M": 64, "TILE_N": 32, "TILE_K": 32, "GROUP_M": 2},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"TILE_M": 64, "TILE_N": 64, "TILE_K": 32, "GROUP_M": 2},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"TILE_M": 128, "TILE_N": 32, "TILE_K": 32, "GROUP_M": 2},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"TILE_M": 128, "TILE_N": 64, "TILE_K": 32, "GROUP_M": 2},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"TILE_M": 128, "TILE_N": 128, "TILE_K": 32, "GROUP_M": 2},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"TILE_M": 32, "TILE_N": 32, "TILE_K": 32, "GROUP_M": 1},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"TILE_M": 64, "TILE_N": 32, "TILE_K": 32, "GROUP_M": 2},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"TILE_M": 64, "TILE_N": 64, "TILE_K": 32, "GROUP_M": 2},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"TILE_M": 128, "TILE_N": 32, "TILE_K": 32, "GROUP_M": 2},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"TILE_M": 128, "TILE_N": 64, "TILE_K": 32, "GROUP_M": 2},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"TILE_M": 128, "TILE_N": 128, "TILE_K": 32, "GROUP_M": 2},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.heuristics(
    {
        "DIVISIBLE_M": lambda args: args["M"] % args["TILE_M"] == 0,
        "DIVISIBLE_N": lambda args: args["N"] % args["TILE_N"] == 0,
        "DIVISIBLE_K": lambda args: args["K"] % args["TILE_K"] == 0,
    }
)
@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    # batch offsets
    pid_b = tl.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N

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

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    num_iters = tl.cdiv(K, TILE_K)
    o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
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

        offs_k += TILE_K
        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

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
    O = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    bmm_kernel[grid_fn](A, B, O, M, N, K)
    return O
