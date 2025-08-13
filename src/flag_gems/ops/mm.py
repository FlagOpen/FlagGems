import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.mm_streamk import streamk_mm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle


@lru_cache(maxsize=1)
def get_device_info():
    try:
        device_id = torch_device_fn.current_device()
    except Exception:
        device_id = 0

    try:
        props = torch_device_fn.get_device_properties(device_id)
        return device_id, props.L2_cache_size, props.multi_processor_count
    except Exception:
        # fallback for A100 default attributes
        # L2 cache size is 40MB and SM count is 108 for A100
        return device_id, 40 * 1024 * 1024, 108


def get_device_id():
    return get_device_info()[0]


def get_l2_cache_size():
    return get_device_info()[1]


def get_sm_count():
    return get_device_info()[2]


CACHE_USAGE_THRESHOLD = 0.8

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


# @libentry()
# @libtuner(
#     #configs=runtime.get_tuned_config("mm_iobound") + runtime.get_tuned_config("mm"),
#     configs=runtime.get_tuned_config("mm"),
#     key=["M", "N", "K", "stride_am", "stride_bk"],
# )
# @triton.heuristics(
#     {
#         "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"]) == 0,
#     }
# )
# @triton.jit
# def mm_kernel_with_grouped_k(
#     A,
#     B,
#     C,  # [Split_K, M, N]
#     M,
#     N,
#     K,
#     stride_am,
#     stride_ak,
#     stride_bk,
#     stride_bn,
#     stride_cb,
#     stride_cm,
#     stride_cn,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
#     BLOCK_K: tl.constexpr,
#     SPLIT_K: tl.constexpr,  # Number of split-K groups
#     GROUP_K_LENGTH: tl.constexpr,
#     EVEN_K: tl.constexpr,
# ):
#     pid = tl.program_id(0)
#     assert GROUP_K_LENGTH % BLOCK_K == 0, "GROUP_K_LENGTH must be divisible by BLOCK_K"

#     num_blocks_m = tl.cdiv(M, BLOCK_M)
#     total_num_m = num_blocks_m * SPLIT_K

#     pid_n = pid // total_num_m
#     odd_column = pid_n % 2
#     pid_m_normal = pid % total_num_m
#     # this is a line-one implementation for the following code:
#     #     if odd_column:
#     #         pid_m_for_c = (total_num_m - 1) - pid_m_normal
#     #     else:
#     #         pid_m_for_c = pid_m_normal
#     pid_m_for_c = (1 - odd_column) * pid_m_normal + odd_column * (
#         total_num_m - 1 - pid_m_normal
#     )

#     pid_m = pid_m_for_c % num_blocks_m
#     pid_k = pid_m_for_c // num_blocks_m

#     # Calculate K_LENGTH based on pid_k
#     group_k_length = min(K - pid_k * GROUP_K_LENGTH, GROUP_K_LENGTH)

#     # matrix multiplication
#     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#     k_start = pid_k * GROUP_K_LENGTH
#     offs_k = k_start + tl.arange(0, BLOCK_K)

#     offs_am = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
#     offs_bn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)

#     # pointers
#     A_ptr = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     B_ptr = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

#     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

#     for k in range(0, tl.cdiv(group_k_length, BLOCK_K)):
#         if EVEN_K:
#             a = tl.load(A_ptr)
#             b = tl.load(B_ptr)
#         else:
#             k_remaining = k_start + group_k_length - k * BLOCK_K
#             a = tl.load(A_ptr, mask=offs_k[None, :] < k_remaining, other=0.0)
#             b = tl.load(B_ptr, mask=offs_k[:, None] < k_remaining, other=0.0)
#         if a.dtype != b.dtype:
#             a = a.to(C.dtype.element_ty)
#             b = b.to(C.dtype.element_ty)
#         acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
#         A_ptr += BLOCK_K * stride_ak
#         B_ptr += BLOCK_K * stride_bk
#     acc = acc.to(C.dtype.element_ty)

#     # Store results
#     offs_cb = pid_k * stride_cb
#     offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

#     C_ptr = C + offs_cb + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
#     mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]

#     tl.store(C_ptr, acc, mask=mask)


# @libentry()
# #@triton.autotune(configs=runtime.get_tuned_config("sum"), key=["M", "N"])
# @triton.jit
# def group_merge_kernel(
#     SRC,  # [SPLIT_K, M, N] 3D Tensor
#     DST,  # [M, N]
#     SPLIT_K,
#     M,
#     N,
#     stride_k,
#     stride_m,
#     stride_n,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
# ):
#     offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
#     offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

#     mask_m = offs_m < M
#     mask_n = offs_n < N

#     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

#     for k in range(SPLIT_K):
#         src_ptr = (
#             SRC + k * stride_k + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
#         )
#         sub_matrix = tl.load(src_ptr, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

#         acc += sub_matrix
#     acc = acc.to(DST.dtype.element_ty)
#     dst_ptr = DST + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
#     tl.store(dst_ptr, acc, mask=mask_m[:, None] & mask_n[None, :])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm_iobound"),
    # Add 'stride_am' and 'stride_bk' to trigger autotune for tensors with the same shape but different strides.
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["log", "log", "log", "log", "log"],
)
@triton.jit
def mm_kernel_iobound(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # column major tile
    pid = tle.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % grid_m
    pid_n = pid // grid_m

    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    prev_multiple = prev_multiple_of(K, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_multiple, BLOCK_K):
        rk = start_k + tl.arange(0, BLOCK_K)
        a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
        b = tl.load(B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn))
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # loop peeling
    rk = prev_multiple + tl.arange(0, BLOCK_K)
    mask_k = rk < K
    a = tl.load(
        A + (ram[:, None] * stride_am + rk[None, :] * stride_ak), mask=mask_k[None, :]
    )
    b = tl.load(
        B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn), mask=mask_k[:, None]
    )
    if a.dtype != b.dtype:
        a = a.to(C.dtype.element_ty)
        b = b.to(C.dtype.element_ty)
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

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
    strategy=["align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def mm_kernel_general(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
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
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    prev_multiple = prev_multiple_of(K, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_multiple, BLOCK_K):
        rk = start_k + tl.arange(0, BLOCK_K)
        a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
        b = tl.load(B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn))
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # loop peeling
    rk = prev_multiple + tl.arange(0, BLOCK_K)
    mask_k = rk < K
    a = tl.load(
        A + (ram[:, None] * stride_am + rk[None, :] * stride_ak), mask=mask_k[None, :]
    )
    b = tl.load(
        B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn), mask=mask_k[:, None]
    )
    if a.dtype != b.dtype:
        a = a.to(C.dtype.element_ty)
        b = b.to(C.dtype.element_ty)
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    tl.store(C, acc, mask=mask)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


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


# def splitk_mm(a, b, c, M, N, K, c_dtype):
#     GROUP_K_LENGTH = 1024
#     SPLIT_K = triton.cdiv(K, GROUP_K_LENGTH)
#     # TODO: float32 or c_dtype
#     multi_c = torch.empty((SPLIT_K, M, N), device=a.device, dtype=c_dtype)
#     # 1st kernel: compute partial results
#     grid = lambda META: (
#         triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]) * SPLIT_K,
#     )
#     grid2 = lambda META: (
#         triton.cdiv(M, META["BLOCK_M"]),
#         triton.cdiv(N, META["BLOCK_N"]),
#     )
#     with torch_device_fn.device(a.device):
#         mm_kernel_with_grouped_k[grid](
#             a,
#             b,
#             multi_c,
#             M,
#             N,
#             K,
#             a.stride(0),
#             a.stride(1),
#             b.stride(0),
#             b.stride(1),
#             multi_c.stride(0),
#             multi_c.stride(1),
#             multi_c.stride(2),
#             SPLIT_K=SPLIT_K,
#             GROUP_K_LENGTH=GROUP_K_LENGTH,
#         )
#         # return torch.sum(multi_c, dim=0)
#         # 2nd kernel: merge partial results
#         group_merge_kernel[grid2](
#             multi_c,
#             c,
#             SPLIT_K,
#             M,
#             N,
#             multi_c.stride(0),
#             multi_c.stride(1),
#             multi_c.stride(2)
#         )
#     return c


def iobound_mm(a, b, c, M, N, K):
    logger.debug(
        "GEMS MM, [mm scenario]: iobound, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s",
        M,
        N,
        K,
        a.stride(0) == 1,
        b.stride(0) == 1,
    )
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_iobound[grid](
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
        )
    return c


def general_mm(a, b, c, M, N, K):
    logger.debug(
        "GEMS MM, [mm scenario]: general, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s",
        M,
        N,
        K,
        a.stride(0) == 1,
        b.stride(0) == 1,
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_general[grid](
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
            GROUP_M=8,
        )
    return c


def mini_mm_scenario(a, b, l2_cache_size=40 * 1024 * 1024, cache_usage_threshold=0.8):
    return (
        a.shape[0] <= 256
        and (a.numel() * a.element_size() + b.shape[0] * b.element_size())
        < l2_cache_size * cache_usage_threshold
    )


def streamk_scenario(a, b, M, N, K):
    # TODO: this my change sometime according to the realbenchmark result
    # Currently, the best configuration for streamk has only been tested on A100(capability[0] > 7).
    # The optimal settings for other devices need to be determined through real testing.
    capability = torch_device_fn.get_device_capability(get_device_info())
    return (
        capability[0] > 7
        and a.dtype in [torch.float16, torch.bfloat16]
        and b.dtype in [torch.float16, torch.bfloat16]
        and K > M * 5
        and K > N * 5
    )


# def two_stages_splitk_mm_scenario(M, N, K):
#     return (M < 32 or N < 32) and (K > M * 10 or K > N * 10)


def mm(a, b):
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    # l2_cache_size = get_l2_cache_size()
    sm_count = get_sm_count()
    if streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, c, M, N, K, sm_count=sm_count)
    else:
        return general_mm(a, b, c, M, N, K)


def mm_out(a, b, *, out):
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # l2_cache_size = get_l2_cache_size()
    sm_count = get_sm_count()
    if streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, out, M, N, K, sm_count=sm_count)
    else:
        return general_mm(a, b, out, M, N, K)
