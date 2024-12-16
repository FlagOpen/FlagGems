import logging

import torch
import triton
import triton.language as tl

# from .. import runtime
# from ..runtime import torch_device_fn
from ..utils import libentry
from ..utils import triton_lang_extension as tle


def heur_even_k(args):
    return args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0


def _init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def compute_bound_autotune_config():
    return [
        # basic configs for compute-bound matmuls
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=5,
            num_warps=2,
        ),
        # good for int8
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=5,
            num_warps=2,
        ),
    ]


def io_bound_autotune_config():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                                "SPLIT_K": 1,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
                    # split_k
                    for split_k in [2, 4, 8, 16]:
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_K": block_k,
                                    "SPLIT_K": split_k,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                                pre_hook=_init_to_zero("C"),
                            )
                        )
    return configs


@triton.autotune(
    configs=io_bound_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def mm_kernel_with_grouped_k(
    A,
    B,
    C,  # [batch * M, N]
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
    SPLIT_K: tl.constexpr,  # Number of split-K groups
    group_k_length: tl.constexpr,
):
    pid = tle.program_id(0)

    #  cal the pid_m  & pid_n to fetch the data
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    # num_blocks_n = tl.cdiv(N, BLOCK_N)
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

    # matrix multiplication
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_k = pid_k * group_k_length + tl.arange(0, BLOCK_K)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    # pointers
    A_ptr = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptr = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)

    for k in range(0, tl.cdiv(group_k_length, BLOCK_K)):
        # TODO: ADD EVEN_K:
        a = tl.load(A_ptr)
        b = tl.load(B_ptr)
        # _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
        # a = tl.load(A_ptr, mask=(offs_k < K)[None, :], other=0)
        # b = tl.load(B_ptr, mask=(offs_k < K)[:, None], other=0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)

    # Store results
    offs_cm = pid_m_for_c * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    C_ptr = C + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask = (offs_cm < (M * SPLIT_K))[:, None] & (offs_cn < N)[None, :]
    tl.store(C_ptr, acc, mask=mask)


@triton.jit
def group_merge_kernel(
    SRC,  # [SPLIT_K * M, N] 2D Tensor
    DST,  # [M, N]
    M,
    N,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    stride_sk = M * N

    # num_blocks_m = pid_m
    # num_blocks_n = tl.cdiv(N, BLOCK_N)

    offs_m = pid_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M * SPLIT_K
    mask_n = offs_n < N

    acc = tl.zeros((1, BLOCK_N), dtype=DST.dtype.element_ty)

    for k in range(SPLIT_K):
        src_ptr = (
            SRC
            + k * stride_sk
            + offs_m[:, None] * stride_dm
            + offs_n[None, :] * stride_sn
        )
        sub_matrix = tl.load(src_ptr, mask=mask_m[:, None] & mask_n[None, :], other=0)

        acc += sub_matrix

    dst_ptr = DST + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
    tl.store(dst_ptr, acc, mask=mask_m[:, None] & mask_n[None, :])


@libentry()
@triton.autotune(
    configs=compute_bound_autotune_config(),
    key=["M", "N", "K"],
)
@triton.heuristics(
    {
        "EVEN_K": heur_even_k,
        "PADDED_M": lambda args: triton.next_power_of_2(args["M"]),
    }
)
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
    PADDED_M: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # matrix multiplication
    pid = tle.program_id(0)
    pid_z = tle.program_id(1)
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
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


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


def mm(a, b):
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    # is_b_column_major = b.stride(0) == 1 and b.stride(1) >= b.size(0)
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    dot_out_dtype = tl.float32
    if M <= 32 & K > 2048:
        logging.debug(
            f"GEMS MM(SPLIT_K MODE), Split-k MODE the input shape(M, N, K) is [{M}, {N}, {K}]"
        )
        BLOCK_M = (
            16 if M <= 16 else 32
        )  # this lead the  triton.cdiv(M, META["BLOCK_M"]) always returns 1.
        BLOCK_N = 32
        BLOCK_K = 16
        GROUP_K = 1024
        # GROUP_K = 1024 #  （we can try 512）
        num_groups_k = triton.cdiv(K, GROUP_K)  # 4
        multi_c = torch.empty((num_groups_k * M, N), device=device, dtype=c_dtype)
        # 1st kernel: compute partial results
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) * num_groups_k,)
        with torch.cuda.device(a.device):
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
                dot_out_dtype=dot_out_dtype,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                SPLIT_K=num_groups_k,
                group_k_length=GROUP_K,
            )

        # 2nd kernel: merge partial results
        grid = (M, triton.cdiv(N, BLOCK_N))
        with torch.cuda.device(a.device):
            group_merge_kernel[grid](
                multi_c,
                c,
                M,
                N,
                multi_c.stride(0),
                multi_c.stride(1),
                c.stride(0),
                c.stride(1),
                dot_out_dtype=dot_out_dtype,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                SPLIT_K=num_groups_k,
            )
        return c

    else:
        logging.debug(f"GEMS MM, the input shape(M, N, K) is [{M}, {N}, {K}]")
        # launch kernel
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            META["SPLIT_K"],
        )
        with torch.cuda.device(a.device):
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
            )
        return c
