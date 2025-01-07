import logging

import torch
import triton
import triton.language as tl

from .. import runtime
from ..runtime import torch_device_fn
from ..utils import libentry, libtuner
from ..utils import triton_lang_extension as tle

try:
    L2_CACHE_SIZE = torch_device_fn.get_device_properties(0).L2_cache_size
except AttributeError:
    L2_CACHE_SIZE = 40 * 1024 * 1024  # 40MB in bytes
CACHE_USAGE_THRESHOLD = 0.7


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_iobound"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
)
@triton.jit
def mm_kernel_with_grouped_k(
    A,
    B,
    C,  # [Split_K, M, N]
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,  # Number of split-K groups
    GROUP_K_LENGTH: tl.constexpr,
):
    pid = tl.program_id(0)
    assert GROUP_K_LENGTH % BLOCK_K == 0, "GROUP_K_LENGTH must be divisible by BLOCK_K"

    num_blocks_m = tl.cdiv(M, BLOCK_M)
    total_num_m = num_blocks_m * SPLIT_K

    pid_n = pid // total_num_m
    odd_column = pid_n % 2
    pid_m_normal = pid % total_num_m
    # this is a line-one implementation for the following code:
    #     if odd_column:
    #         pid_m_for_c = (total_num_m - 1) - pid_m_normal
    #     else:
    #         pid_m_for_c = pid_m_normal
    pid_m_for_c = (1 - odd_column) * pid_m_normal + odd_column * (
        total_num_m - 1 - pid_m_normal
    )

    pid_m = pid_m_for_c % num_blocks_m
    pid_k = pid_m_for_c // num_blocks_m

    # Calculate K_LENGTH based on pid_k
    group_k_length = min(K - pid_k * GROUP_K_LENGTH, GROUP_K_LENGTH)

    # matrix multiplication
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_start = pid_k * GROUP_K_LENGTH
    offs_k = k_start + tl.arange(0, BLOCK_K)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    # pointers
    A_ptr = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptr = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)

    for k in range(0, tl.cdiv(group_k_length, BLOCK_K)):  #
        k_remaining = k_start + group_k_length - k * BLOCK_K
        # TODO: ADD EVEN_K:
        a = tl.load(A_ptr, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(B_ptr, mask=offs_k[:, None] < k_remaining, other=0.0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)

    # Store results
    offs_cb = pid_k * stride_cb
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    C_ptr = C + offs_cb + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]

    tl.store(C_ptr, acc, mask=mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("sum"), key=["M", "N"])
@triton.jit
def group_merge_kernel(
    SRC,  # [SPLIT_K, M, N] 3D Tensor
    DST,  # [M, N]
    SPLIT_K,
    M,
    N,
    stride_k,
    stride_m,
    stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(SPLIT_K):
        src_ptr = (
            SRC + k * stride_k + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        )
        sub_matrix = tl.load(src_ptr, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        acc += sub_matrix
    acc = acc.to(DST.dtype.element_ty)
    dst_ptr = DST + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    tl.store(dst_ptr, acc, mask=mask_m[:, None] & mask_n[None, :])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_iobound"),
    # configs=runtime.get_tuned_config("mm_iobound") + runtime.get_tuned_config("mm"),
    # Add 'stride_am' and 'stride_bk' to trigger autotune for tensors with the same shape but different strides.
    key=["M", "N", "K", "stride_am", "stride_bk"],
)
@triton.heuristics(runtime.get_heuristic_config("mm"))
@triton.jit
def iobound_mm_kernel(
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
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    b_column_major: tl.constexpr,
):
    # matrix multiplication
    pid = tle.program_id(0)

    grid_m = tl.cdiv(M, BLOCK_M)

    pid_m = pid % grid_m
    pid_n = pid // grid_m

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * BLOCK_K
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    tl.store(C, acc, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    # Add 'stride_am' and 'stride_bk' to trigger autotune for tensors with the same shape but different strides.
    key=["M", "N", "K", "stride_am", "stride_bk"],
)
@triton.heuristics(runtime.get_heuristic_config("mm"))
@triton.jit
def mm_kernel(
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
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    b_column_major: tl.constexpr,
):
    # matrix multiplication
    pid = tle.program_id(0)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * BLOCK_K
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    tl.store(C, acc, mask=mask)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def mini_matrix_scenario(
    a, b, l2_cache_size=40 * 1024 * 1024, cache_usage_threshold=0.8
):
    return (
        a.shape[0] <= 256
        and (a.numel() * a.element_size() + b.shape[0] * b.element_size())
        < l2_cache_size * cache_usage_threshold
    )


def largek_mm_scenario(M, N, K):
    return False
    # return K > N * 10 and K > M * 10


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def splitk_mm(a, b, c, M, N, K, c_dtype, dot_out_dtype, b_column_major_flag):
    logging.debug("GEMS MM (SPLITK)")
    GROUP_K_LENGTH = 1024
    SPLIT_K = triton.cdiv(K, GROUP_K_LENGTH)
    # TODO: float32 or c_dtype
    multi_c = torch.empty((SPLIT_K, M, N), device=a.device, dtype=c_dtype)
    # 1st kernel: compute partial results
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]) * SPLIT_K,
    )
    with torch_device_fn.device(a.device):
        mm_kernel_with_grouped_k[grid](
            a,
            b,
            multi_c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            multi_c.stride(0),
            multi_c.stride(1),
            multi_c.stride(2),
            dot_out_dtype=dot_out_dtype,
            SPLIT_K=SPLIT_K,
            GROUP_K_LENGTH=GROUP_K_LENGTH,
        )

    # 2nd kernel: merge partial results
    grid2 = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        group_merge_kernel[grid2](
            multi_c,
            c,
            SPLIT_K,
            M,
            N,
            multi_c.stride(0),
            multi_c.stride(1),
            multi_c.stride(2),
            dot_out_dtype=dot_out_dtype,
        )
    return c


def iobound_mm(a, b, c, M, N, K, dot_out_dtype, b_column_major_flag):
    logging.debug("GEMS MM (iobound)")
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        iobound_mm_kernel[grid](
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
            dot_out_dtype=dot_out_dtype,
            GROUP_M=8,
            b_column_major=b_column_major_flag,
        )
    return c


def general_mm(a, b, c, M, N, K, dot_out_dtype, b_column_major_flag):
    logging.debug("GEMS MM (general)")
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel[grid](
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
            dot_out_dtype=dot_out_dtype,
            GROUP_M=8,
            b_column_major=b_column_major_flag,
        )
    return c


def mm(a, b):
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    b_column_major_flag = b.stride(0) == 1 and b.stride(1) == b.size(0)
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    dot_out_dtype = tl.float32
    if b_column_major_flag and mini_matrix_scenario(
        a, b, L2_CACHE_SIZE, CACHE_USAGE_THRESHOLD
    ):
        return iobound_mm(a, b, c, M, N, K, dot_out_dtype, b_column_major_flag)
    elif largek_mm_scenario(M, N, K):
        return splitk_mm(a, b, c, M, N, K, c_dtype, dot_out_dtype, b_column_major_flag)
    else:
        return general_mm(a, b, c, M, N, K, dot_out_dtype, b_column_major_flag)
