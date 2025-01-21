import logging

import torch
import triton
import triton.language as tl


@triton.jit()
def swizzle_tile(
    tile_id,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n


@triton.jit(
    do_not_specialize=["full", "remaining", "iters_per_tile", "start_iter", "end_iter"]
)
def mac_loop(
    A,
    B,
    C,
    P,
    M,
    N,
    K,
    locks,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    full,
    remaining,
    iters_per_tile,
    start_iter,
    end_iter,
    ACC_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # where are we in the grid
    pid = tl.program_id(0)
    tile_id = start_iter // iters_per_tile

    pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    if stride_am == 1:
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if stride_bk == 1:
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    # pointers
    A_ptr = (
        A
        + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        + BLOCK_K * stride_ak * (start_iter % iters_per_tile)
    )
    B_ptr = (
        B
        + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
        + BLOCK_K * stride_bk * (start_iter % iters_per_tile)
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for current_iter in range(start_iter, end_iter):
        # TODO: when k is not even, we need to load the data from A and B with mask.
        a = tl.load(A_ptr)
        b = tl.load(B_ptr)
        # acc += tl.dot(a, b)
        acc += tl.dot(a, b, out_dtype=ACC_TYPE, allow_tf32=False)
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk

    rm1 = tl.arange(0, BLOCK_M)
    rn1 = tl.arange(0, BLOCK_N)

    # the first situation: not the starting parts. only need to store the data on P
    if start_iter % iters_per_tile != 0:
        P_ptr = P + pid * BLOCK_M * BLOCK_N + (rm1[:, None] * BLOCK_N + rn1[None, :])
        tl.store(P_ptr, acc, cache_modifier=".cg")
        # tl.debug_barrier()
        tl.atomic_xchg(locks + pid, 1)
    else:  # the first part of certain grids. shoud read datas and merge datas
        next_pid = pid + 1
        stop_loading_iter = start_iter + iters_per_tile
        end = end_iter
        while end < stop_loading_iter:
            while tl.atomic_cas(locks + next_pid, 1, 1) != 1:
                pass
            P_ptr = (
                P
                + next_pid * BLOCK_M * BLOCK_N
                + (rm1[:, None] * BLOCK_N + rn1[None, :])
            )
            acc += tl.load(P_ptr, cache_modifier=".cg")
            end += full + (next_pid < remaining)
            next_pid += 1

        # acc = acc.to(C.dtype.element_ty)  #
        C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, acc, mask=mask)


@triton.jit()
def first_wave(
    A,
    B,
    C,
    P,
    M,
    N,
    K,
    locks,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    full,
    remaining,
    iters_per_tile,
    ACC_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)  # FROM 0 TO SM_COUNT
    start_iter = pid * full + tl.minimum(pid, remaining)
    last_iter = (pid + 1) * full + tl.minimum(pid + 1, remaining)

    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        # iterate over K axis, M/N may change during iteration, so we need to re-calculate the end_iter
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        mac_loop(
            A,
            B,
            C,
            P,
            M,
            N,
            K,
            locks,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            full,
            remaining,
            iters_per_tile,
            start_iter,
            end_iter,
            ACC_TYPE,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            GROUP_M,
        )
        start_iter = end_iter


@triton.jit()
def classic_mm(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    total_tiles_streamk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # first wave has done more tiles than there are SMs, we adjust pid
    tile_id = tl.program_id(0) + total_tiles_streamk

    pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    if stride_am == 1:
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if stride_bk == 1:
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A)
        b = tl.load(B)
        # acc += tl.dot(a, b)
        acc += tl.dot(a, b, out_dtype=ACC_TYPE, allow_tf32=False)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(C, acc)


def streamk_mm(a, b, c, M, N, K, c_dtype, acc_type, sm_count=108):
    # TODO: change the hard code to tuning config
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
    num_stages = 3
    num_warps = 8

    GROUP_M = 8
    number_blocks_m = triton.cdiv(M, BLOCK_M)
    number_blocks_n = triton.cdiv(N, BLOCK_N)

    total_tiles = number_blocks_m * number_blocks_n
    iters_per_tile = triton.cdiv(K, BLOCK_K)
    tiles_per_wave = sm_count

    number_cooperative_tiles = total_tiles % tiles_per_wave
    # mini wave
    total_iters_streamk = number_cooperative_tiles * iters_per_tile
    tiles_per_pid = total_iters_streamk // tiles_per_wave
    tile_remaining = total_iters_streamk % tiles_per_wave

    locks = torch.zeros((tiles_per_wave,), device=a.device, dtype=torch.int32)
    P = torch.zeros(
        (tiles_per_wave, BLOCK_M, BLOCK_N), device=a.device, dtype=torch.float32
    )
    # with torch_device_fn.device(a.device):
    k1 = first_wave[(tiles_per_wave,)](
        a,
        b,
        c,
        P,
        M,
        N,
        K,
        locks,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        full=tiles_per_pid,
        remaining=tile_remaining,
        iters_per_tile=iters_per_tile,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        ACC_TYPE=acc_type,
        GROUP_M=GROUP_M,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    logging.DEBUG(f"{k1.n_regs} registers used, {k1.n_spills} spills")
    logging.DEBUG(f"shared memory: {k1.metadata.shared} bytes")

    k2 = classic_mm[(total_tiles - number_cooperative_tiles,)](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        total_tiles_streamk=number_cooperative_tiles,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        ACC_TYPE=acc_type,
        GROUP_M=GROUP_M,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    logging.DEBUG(f"{k2.n_regs} registers used, {k2.n_spills} spills")
    logging.DEBUG(f"shared memory: {k2.metadata.shared} bytes")
    return c
