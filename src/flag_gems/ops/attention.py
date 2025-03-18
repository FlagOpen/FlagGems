import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import update_philox_state

from .. import runtime


# Modified from Triton tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    mask_block_ptr,  #
    stride_k_seqlen,
    stride_v_seqlen,
    stride_attn_mask_kv_seqlen,  #
    start_m,
    qk_scale,  #
    q_load_mask,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    KV_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    # causal = False
    else:
        lo, hi = 0, KV_CTX

    K_block_ptr += lo * stride_k_seqlen
    V_block_ptr += lo * stride_v_seqlen
    kv_load_mask = lo + offs_n < KV_CTX
    if HAS_ATTN_MASK:
        mask_block_ptr += lo * stride_attn_mask_kv_seqlen

    LOG2E: tl.constexpr = 1.44269504

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        # start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr, mask=kv_load_mask[None, :], other=0.0)
        if PRE_LOAD_V:
            v = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)

        qk = tl.dot(q, k, allow_tf32=False)
        # qk = qk.to(tl.float32)

        if HAS_ATTN_MASK:
            attn_mask = tl.load(
                mask_block_ptr,
                mask=q_load_mask[:, None] & kv_load_mask[None, :],
                other=0.0,
            )

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])

            if HAS_ATTN_MASK:
                qk = qk * qk_scale + attn_mask
                qk *= LOG2E
                qk = qk + tl.where(mask, 0, -1.0e6)
            else:
                qk = qk * qk_scale * LOG2E + tl.where(mask, 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            if HAS_ATTN_MASK:
                qk = qk * qk_scale + attn_mask
                qk *= LOG2E
                qk = qk - m_ij[:, None]
            else:
                qk = qk * qk_scale * LOG2E - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        if not PRE_LOAD_V:
            v = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(q.dtype)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc, allow_tf32=False)
        # update m_i and l_i
        m_i = m_ij

        K_block_ptr += BLOCK_N * stride_k_seqlen
        V_block_ptr += BLOCK_N * stride_v_seqlen

        if HAS_ATTN_MASK:
            mask_block_ptr += BLOCK_N * stride_attn_mask_kv_seqlen

    return acc, l_i, m_i


def early_config_prune(configs, nargs, **kwargs):
    return list(filter(lambda cfg: cfg.kwargs["BLOCK_N"] <= nargs["HEAD_DIM"], configs))


@triton.autotune(
    configs=runtime.get_tuned_config("attention"),
    key=["KV_CTX", "HEAD_DIM"],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": None,
        "top_k": 1.0,
    },
)
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    attn_mask,
    sm_scale,
    Out,  #
    stride_q_batch,
    stride_q_head,
    stride_q_seqlen,
    stride_q_headsize,
    stride_k_batch,
    stride_k_head,
    stride_k_seqlen,
    stride_k_headsize,
    stride_v_batch,
    stride_v_head,
    stride_v_seqlen,
    stride_v_headsize,
    stride_attn_mask_batch,
    stride_attn_mask_head,
    stride_attn_mask_q_seqlen,
    stride_attn_mask_kv_seqlen,
    stride_o_batch,
    stride_o_head,
    stride_o_seqlen,
    stride_o_headsize,
    Z,
    q_numhead,
    kv_numhead,
    Q_CTX,
    KV_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_id = off_hz // q_numhead
    head_id = off_hz % q_numhead
    kv_head_id = off_hz % kv_numhead

    q_offset = (
        batch_id.to(tl.int64) * stride_q_batch + head_id.to(tl.int64) * stride_q_head
    )
    kv_offset = (
        batch_id.to(tl.int64) * stride_k_batch + kv_head_id.to(tl.int64) * stride_k_head
    )

    offs_headsize = tl.arange(0, HEAD_DIM)

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_load_mask = offs_m < Q_CTX
    offs_n = tl.arange(0, BLOCK_N)

    Q_block_ptr = (
        Q
        + q_offset
        + offs_m[:, None] * stride_q_seqlen
        + offs_headsize[None, :] * stride_q_headsize
    )
    K_block_ptr = (
        K
        + kv_offset
        + offs_n[None, :] * stride_k_seqlen
        + offs_headsize[:, None] * stride_k_headsize
    )
    V_block_ptr = (
        V
        + kv_offset
        + offs_n[:, None] * stride_v_seqlen
        + offs_headsize[None, :] * stride_v_headsize
    )

    if HAS_ATTN_MASK:
        attn_mask_offset = (
            batch_id.to(tl.int64) * stride_attn_mask_batch
            + head_id.to(tl.int64) * stride_attn_mask_head
        )
        mask_block_ptr = (
            attn_mask
            + attn_mask_offset
            + offs_m[:, None] * stride_attn_mask_q_seqlen
            + offs_n[None, :] * stride_attn_mask_kv_seqlen
        )
    else:
        mask_block_ptr = None

    O_block_ptr = (
        Out
        + q_offset
        + offs_m[:, None] * stride_o_seqlen
        + offs_headsize[None, :] * stride_o_headsize
    )

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    # qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, mask=q_load_mask[:, None], other=0.0)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            mask_block_ptr,
            stride_k_seqlen,
            stride_v_seqlen,
            stride_attn_mask_kv_seqlen,
            start_m,
            qk_scale,
            q_load_mask,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            KV_CTX,
            V.dtype.element_ty == tl.float8e5,
            HAS_ATTN_MASK,
            PRE_LOAD_V,
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            mask_block_ptr,
            stride_k_seqlen,
            stride_v_seqlen,
            stride_attn_mask_kv_seqlen,
            start_m,
            qk_scale,
            q_load_mask,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            KV_CTX,
            V.dtype.element_ty == tl.float8e5,
            HAS_ATTN_MASK,
            PRE_LOAD_V,
        )
    # epilogue
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=q_load_mask[:, None])


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    logging.debug("GEMS SCALED DOT PRODUCT ATTENTION")
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    assert dropout_p == 0.0, "Currenty only support dropout_p=0.0"

    o = torch.empty_like(query, dtype=value.dtype)

    stage = 3 if is_causal else 1

    if scale is None:
        sm_scale = 1.0 / (HEAD_DIM_K**0.5)
    else:
        sm_scale = scale

    kv_head_num = key.shape[1]

    grid = lambda args: (
        triton.cdiv(query.shape[2], args["BLOCK_M"]),
        query.shape[0] * query.shape[1],
        1,
    )

    if attn_mask is not None:
        HAS_ATTN_MASK = True
        stride_attn_mask_batch = attn_mask.stride(0)
        stride_attn_mask_head = attn_mask.stride(1)
        stride_attn_mask_q_seqlen = attn_mask.stride(2)
        stride_attn_mask_kv_seqlen = attn_mask.stride(3)
    else:
        HAS_ATTN_MASK = False
        stride_attn_mask_batch = 1
        stride_attn_mask_head = 1
        stride_attn_mask_q_seqlen = 1
        stride_attn_mask_kv_seqlen = 1

    with torch_device_fn.device(query.device):
        _attn_fwd[grid](
            query,
            key,
            value,
            attn_mask,
            sm_scale,
            o,  #
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),  #
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),  #
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),  #
            stride_attn_mask_batch,
            stride_attn_mask_head,
            stride_attn_mask_q_seqlen,
            stride_attn_mask_kv_seqlen,  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            query.shape[0],
            query.shape[1],
            kv_head_num,  #
            query.shape[2],  #
            key.shape[2],  #
            HEAD_DIM_K,  #
            STAGE=stage,  #
            HAS_ATTN_MASK=HAS_ATTN_MASK,  #
        )
        return o


# The following implementation is a fundamentally a triton rewrite of TriDao's Flash Attention in Cuda.

@triton.jit
def philox_offset_one_warp(b, h, nh: tl.constexpr):
    # To align with TriDao's implementation, philox_offset linearly determined by
    # a 3d dense tensor (batch_id, head_id, thread_id) with shape (batch_size, num_heads, 32)
    # and stride ( num_heads * 32, 32, 1 )
    return (b * nh + h) * 32 + tl.arange(0, 32)


@triton.jit
def u64_to_lohi(x):
    return (x >> 32).to(tl.uint32), (x & 0xFFFFFFFF).to(tl.uint32)


@triton.jit
def u64_from_lohi(lo, hi):
    return hi.to(tl.uint64) << 32 + lo.to(tl.uint64)


@triton.jit
def philox_(seed, subsequence, offset):
    kPhilox10A: tl.constexpr = 0x9E3779B9
    kPhilox10B: tl.constexpr = 0xBB67AE85
    k0, k1 = u64_to_lohi(seed.to(tl.uint64))
    c0, c1 = u64_to_lohi(offset.to(tl.uint64))
    c2, c3 = u64_to_lohi(subsequence(tl.uint64))

    # pragma unroll
    kPhiloxSA: tl.constexpr = 0xD2511F53
    kPhiloxSB: tl.constexpr = 0xCD9E8D57
    for _ in range(6):
        res0 = kPhiloxSA.to(tl.uint64) * c0.to(tl.uint64)
        res1 = kPhiloxSB.to(tl.uint64) * c2.to(tl.uint64)
        res0_x, res0_y = u64_to_lohi(res0)
        res1_x, res1_y = u64_to_lohi(res1)
        c0, c1, c2, c3 = res1_y ^ c1 ^ k0, res1_x, res0_y ^ c3 ^ k1, res0_x
        k0 += kPhilox10A
        k1 += kPhilox10B

    res0 = kPhiloxSA.to(tl.uint64) * c0.to(tl.uint64)
    res1 = kPhiloxSB.to(tl.uint64) * c2.to(tl.uint64)
    res0_x.res0_y = u64_to_lohi(res0)
    res1_x.res1_y = u64_to_lohi(res1)
    c0, c1, c2, c3 = res1_y ^ c1 ^ k0, res1_x, res0_y ^ c3 ^ k1, res0_x

    return c0, c1, c2, c3


@triton.jit
def apply_dropout_mask(
    P,
    mask,
    encode_dropout_in_sign_bit: tl.constexpr,
):
    if encode_dropout_in_sign_bit:
        P = tl.where(mask, -P, P)
    else:
        P = tl.where(mask, 0, P)
    return P


@triton.jit
def make_4x_dropout_mask(r_u32, p_u8, M: tl.constexpr, N: tl.constexpr):
    r = r_u32
    p = p_u8
    m0 = tl.where(r & 0xFF < p, 0, 1)
    r >>= 8
    m1 = tl.where(r & 0xFF < p, 0, 1)
    m0 = tl.join(m0, m1).trans(2, 0, 1).reshape(2 * M, N)

    r >>= 8
    m0 = tl.where(r & 0xFF < p, 0, 1)
    r >>= 8
    m1 = tl.where(r & 0xFF < p, 0, 1)
    m1 = tl.join(m0, m1).trans(2, 0, 1).reshape(2 * M, N)

    m = tl.join(m0, m1).trans(2, 0, 1).reshape(4 * M, N)
    return m


@triton.jit(
    do_not_specialize=[
        "b",
        "h",
        "row_start",
        "col_start",
        "philox_seed",
        "philox_offset",
    ]
)
def apply_dropout(
    P,
    sor,
    soc,
    bid,
    hid,
    philox_seed,
    philox_offset,
    p_dropout_uint8: tl.constexpr,
    encode_dropout_in_sign_bit: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # P is of size (BLOCK_M, BLOCK_N) and its scalar bitsize is 32
    # BLOCK_M is ensured to be a multiple of 16, BLOCK_N a multiple of 32
    M: tl.constexpr = BLOCK_M // 16
    N: tl.constexpr = BLOCK_N // 32
    row = sor + tl.arange(0, M)[:, None]
    col = soc + tl.arange(0, BLOCK_N)[None, :] // 32

    tid = tl.arange(0, BLOCK_N)[None, :] % 32
    philox_offset += (bid * NUM_HEADS + hid) * 32 + tid

    subsequence = u64_from_lohi(row * 32, col)
    r0, r1, r2, r3 = philox_(philox_seed, subsequence, philox_offset)

    # Fully unrolled due to triton's inability to concat 2d tensor
    m0 = make_4x_dropout_mask(r0, p_dropout_uint8, M, N)
    m1 = make_4x_dropout_mask(r1, p_dropout_uint8, M, N)
    m0 = tl.join(m0, m1).trans(2, 0, 1).reshape(8 * M, N)

    m0 = make_4x_dropout_mask(r0, p_dropout_uint8, M, N)
    m1 = make_4x_dropout_mask(r1, p_dropout_uint8, M, N)
    m1 = tl.join(m0, m1).trans(2, 0, 1).reshape(8 * M, N)

    m = tl.join(m0, m1).trans(2, 0, 1).reshape(16 * M, N)
    P = apply_dropout_mask(P, m)
    return P


@triton.jit(do_not_specialize=['max_seqlen_q', 'max_seqlen_k'])
def apply_mask(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    ws_left,
    ws_right,
    alibi_slope,
    is_even_mn: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    has_alibi: tl.constexpr,
):
    need_mask: tl.constexpr = is_causal | has_alibi | is_local | (not is_even_mn)
    if need_mask:
        col_lb = max(0, row_idx + max_seqlen_k - max_seqlen_q - ws_left)
        col_rb = min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + ws_right)

        if not has_alibi:
            alibi_slope = .0

        S -= alibi_slope * tl.abs(col_idx[None, :] - row_idx[:, None])

        if is_causal:
            S = tl.where(col_idx[None, :] >= col_rb[:, None], float('-inf'), S)

        if is_local:
            S = tl.where(col_idx[None, :] >= col_rb[:, None] | col_idx[None, :] < col_lb[:, None], float('-inf'), S)
        
        if (not is_local) & (not is_causal) & (not is_even_mn):
            S = tl.where(col_idx[None, :] >= max_seqlen_k, float('-inf'), S)

    return S


@triton.jit
def softmax_rescale(
    O_acc,
    S,
    row_max,
    row_sum,
    softmax_scale_log2e: tl.constexpr,
    is_border: tl.constexpr,
    # is_init: tl.constexpr
):
    prev_max = row_max
    row_max = tl.maximum(row_max, tl.max(S, 1))

    # if not is_init:
    #     if is_border:
    #         cur_max = tl.where(row_max == float('-inf'), 0, row_max)
    #     else:
    #         cur_max = row_max
    #     p_scale = tl.math.exp2((prev_max - cur_max) * softmax_scale_log2e)
    #     row_sum *= p_scale
    #     O_acc *= p_scale[:, None]

    if is_border:
        cur_max = tl.where(row_max == float('-inf'), 0, row_max)
    else:
        cur_max = row_max
    p_scale = tl.math.exp2((prev_max - cur_max) * softmax_scale_log2e)
    row_sum *= p_scale
    O_acc *= p_scale[:, None]

    max_scaled = tl.where(row_max == float('-inf'), 0, row_max * softmax_scale_log2e)
    P = tl.math.exp2(S * softmax_scale_log2e - max_scaled[:, None])
    row_sum = row_sum + tl.sum(P, 1)
    return O_acc, P, row_max, row_sum


def block_m_heuristic(headdim, is_dropout):
    # return 128 if headdim <= 128 else 64
    return 64

def block_n_heuristic(headdim, is_dropout):
    # return 128 if headdim <= 128 else 64
    return 64

def block_m_splitkv_heuristic(headdim):
    return 128 if headdim <= 128 else 64

def block_n_splitkv_heuristic(headdim):
    if headdim <= 64:
        return 256
    elif headdim <= 128:
        return 128
    else:
        return 64

# @triton.autotune(
#     configs=runtime.get_tuned_config("attention"),
#     key=["HEAD_DIM"],
#     prune_configs_by={
#         "early_config_prune": early_config_prune,
#         "perf_model": None,
#         "top_k": 1.0,
#     },
# )
@triton.heuristics(
    values={
        'BLOCK_M': lambda args: block_m_heuristic(args["HEAD_DIM"], args["is_dropout"]),
        'BLOCK_N': lambda args: block_n_heuristic(args["HEAD_DIM"], args["is_dropout"]),
        'num_warps': lambda args: 4,
        'num_stages': lambda args: 3,
        'PRE_LOAD_V': lambda args: True,
        'IS_EVEN_MN': lambda args: (args["seqlen_q"] % args["BLOCK_M"] == 0) and (args["seqlen_k"] % args["BLOCK_N"] == 0),
    }
)
@triton.jit(do_not_specialize=["seqlen_q", "seqlen_k", "q_b_stride", "k_b_stride", "o_b_stride", "philox_seed", "philox_offset", "pdrop_u8", "slopes_batch_stride"])
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    P_ptr,
    O_ptr,
    lse_ptr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    q_b_stride,
    q_s_stride,
    q_h_stride,
    k_b_stride,
    k_s_stride,
    k_h_stride,
    o_b_stride,
    o_s_stride,
    o_h_stride,
    h,
    hk,
    pSlopes,
    philox_seed,
    philox_offset,
    pdrop_u8,
    rpdrop,
    slopes_batch_stride,
    HEAD_DIM: tl.constexpr,
    is_dropout: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    has_alibi: tl.constexpr,
    softmax_scale: tl.constexpr,
    softmax_scale_log2e: tl.constexpr,
    ws_left: tl.constexpr,
    ws_right: tl.constexpr,
    return_P: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_HEADS_K: tl.constexpr,
    IS_EVEN_MN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_splits: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr
):
    m_block = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)

    if is_local:
        col_min = m_block * BLOCK_M + seqlen_k - seqlen_q - ws_left
        col_min = max(col_min, 0)
    else:
        col_min = 0
    
    col_max = tl.cdiv(seqlen_k, BLOCK_N) * BLOCK_N
    if is_causal or is_local:
        col_max = min(col_max, (m_block + 1) * BLOCK_M + seqlen_k - seqlen_q + ws_right)

    # if is_local:
    #     n_block_min = max(0, (m_block * BLOCK_M + seqlen_k - seqlen_q - ws_left) / BLOCK_N)
    # else:
    #     n_block_min = 0 

    # n_block_max = tl.cdiv(seqlen_k, BLOCK_N)
    
    # if is_causal or is_local:
    #     n_block_max = min(n_block_max,
    #                       tl.cdiv((m_block + 1) * BLOCK_M + seqlen_k - seqlen_q + ws_right, BLOCK_N))

    if has_alibi:
        alibi_offset = bid * slopes_batch_stride + hid
        alibi_slope = tl.load(pSlopes + alibi_offset)
        alibi_slope /= scale
    else:
        alibi_slope = 0.0

    if (not is_causal) and (not is_local):
        if IS_EVEN_MN:
            n_masking_blocks: tl.constexpr = 0
        else:
            n_masking_blocks: tl.constexpr = 1
    elif is_causal and IS_EVEN_MN: # causal implies ws_right is zero
        n_masking_blocks: tl.constexpr = tl.cdiv(BLOCK_M, BLOCK_N)
    else:
        # local and not causal, 
        n_masking_blocks: tl.constexpr = tl.cdiv(BLOCK_M, BLOCK_N) + 1
    
    masking_cols = n_masking_blocks * BLOCK_N

    Q_ptr += bid * q_b_stride
    Q_ptr += hid * q_h_stride
    row_start = m_block * BLOCK_M
    row_idx = row_start + tl.arange(0, BLOCK_M)
    Q_off = row_idx[:, None] * q_s_stride + tl.arange(0, HEAD_DIM)[None, :]
    qmask = row_idx[:, None] < seqlen_q
    if IS_EVEN_MN:
        Q = tl.load(Q_ptr + Q_off, cache_modifier='.cg')
    else:
        Q = tl.load(Q_ptr + Q_off, mask=qmask, cache_modifier='.cg')
    
    if return_P:
        P_ptr += ((bid * NUM_HEADS + hid) * seqlen_q_rounded + m_block * BLOCK_M) * seqlen_k_rounded
        P_ptr += (n_block_max - 1) * BLOCK_N
        P_offset = tl.arange(0, BLOCK_M)[:, None] * seqlen_k_rounded + tl.arange(0, BLOCK_N)

    O_ = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Start from the right most block
    n_block = n_block_max - 1

    h_hk_ratio = h // hk
    K_ptr += bid * k_b_stride
    K_ptr += (hid // h_hk_ratio) * k_h_stride
    V_ptr += bid * k_b_stride
    V_ptr += (hid // h_hk_ratio) * k_h_stride

    K_offset = tl.arange(0, BLOCK_N)[None, :] * k_s_stride + tl.arange(0, HEAD_DIM)[:, None]
    V_offset = tl.arange(0, BLOCK_N)[:, None] * k_s_stride + tl.arange(0, HEAD_DIM)[None, :]

    p_bk0 = K_ptr + K_offset
    p_bv0 = V_ptr + V_offset

    for col_start in tl.range(max(col_min, col_max - masking_cols), col_max, step=BLOCK_N):
    # for r_blk_idx in tl.range(0, min(n_masking_blocks, n_blocks_max - n_blocks_min)):
        off = col_start * k_s_stride
        if IS_EVEN_MN:
            K = tl.load(p_bk0 + off, cache_modifier=".cg")
            if PRE_LOAD_V:
                V = tl.load(p_bv0 + off, cache_modifier=".cg")
        else:
            kvmask = col < seqlen_k
            K = tl.load(p_bk0 + off, mask=kvmask[None, :], cache_modifier=".cg")
            if PRE_LOAD_V:
                V = tl.load(p_bv0 + off, mask=kvmask[:, None], cache_modifier=".cg")
        S = tl.dot(Q, K, allow_tf32=False)
        col_idx = col_start + tl.arange(0, BLOCK_N)
        row_idx = row_start + tl.arange(0, BLOCK_M)
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            seqlen_q,
            seqlen_k,
            ws_left,
            ws_right,
            alibi_slope,
            is_even_mn=IS_EVEN_MN,
            is_causal=is_causal,
            is_local=is_local,
            has_alibi=has_alibi
        )

        # is_init = (n_block == n_block_max - 1).to(tl.int1)
        O_, P, rowmax_, rowsum_ = softmax_rescale(
            O_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=softmax_scale_log2e,
            is_border=(is_causal or is_local),
            # is_init=is_init
        )
        P = P.to(O_ptr.type.element_ty)

        if is_dropout:
            row_start = m_block * (BLOCK_M // 16)
            col_start = n_block * (BLOCK_N // 32)

            if return_P:
                P_drop = P
                P_drop = apply_dropout(
                    P_drop,
                    row_start,
                    col_start,
                    bid,
                    hid,
                    philox_seed,
                    philox_offset,
                    pdrop_u8,
                    encode_dropout_in_sign_bit=True,
                    NUM_HEADS=NUM_HEADS,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                )
                tl.store(P_ptr + P_offset, P_drop, mask=qmask & kvmask[None, :])
                P_offset += BLOCK_N

            P = apply_dropout(
                P,
                row_start,
                col_start,
                bid,
                hid,
                philox_seed,
                philox_offset,
                pdrop_u8,
                encode_dropout_in_sign_bit=False,
                NUM_HEADS=NUM_HEADS,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        if not PRE_LOAD_V:
            off = col_start * k_s_stride
            if IS_EVEN_MN:
                V = tl.load(p_bv0 + off, cache_modifier=".cg")
            else:
                V = tl.load(p_bv0 + off, mask=kvmask[:, None], cache_modifier=".cg")
        O_ = tl.dot(P, V, O_, allow_tf32=False)

    for col_start in tl.range(min_col, max_col - masking_cols, step=BLOCK_N, num_stages=num_stages):
    # for r_blk_idx in tl.range(n_block_max - n_masking_blocks - 1, n_block_min - 1, step=-1, num_stages=num_stages):
        off = col_start * k_s_stride
        K = tl.load(p_bk0 + off, cache_modifier=".cg")
        # if PRE_LOAD_V:
        #     V = tl.load(V_ptr + V_offset, cache_modifier=".cg")
        S = tl.dot(Q, K)

        if PRE_LOAD_V:
            V = tl.load(p_bv0 + off, cache_modifier=".cg")

        col_idx = col_start + tl.arange(0, BLOCK_N)
        row_idx = row_start + tl.arange(0, BLOCK_M)
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            seqlen_q,
            seqlen_k,
            ws_left,
            ws_right,
            alibi_slope,
            is_even_mn=True,
            is_causal=False,
            is_local=is_local,
            has_alibi=has_alibi
        )

        # is_init = (n_block == n_block_max - 1).to(tl.int1)
        O_, P, rowmax_, rowsum_ = softmax_rescale(
            O_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=softmax_scale_log2e,
            is_border=is_local,
        )
        P = P.to(O_ptr.type.element_ty)

        if is_dropout:
            row_start = m_block * (BLOCK_M // 16)
            col_start = n_block * (BLOCK_N // 32)
        
            if return_P:
                P_drop = P
                P_drop = apply_dropout(
                    P_drop,
                    row_start,
                    col_start,
                    bid,
                    hid,
                    philox_seed,
                    philox_offset,
                    pdrop_u8,
                    encode_dropout_in_sign_bit=True,
                    NUM_HEADS=NUM_HEADS,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                )
                tl.store(P_ptr + P_offset, P_drop, mask=qmask & kvmask)
                P_offset += BLOCK_N

            P = apply_dropout(
                P,
                row_start,
                col_start,
                bid,
                hid,
                philox_seed,
                philox_offset,
                pdrop_u8,
                encode_dropout_in_sign_bit=False,
                NUM_HEADS=NUM_HEADS,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        if not PRE_LOAD_V:
            off = col_start * k_s_stride
            V = tl.load(p_bv0 + off, cache_modifier=".cg")

        O_ = tl.dot(P, V, O_)

    # LSE
    # Note, rowsum = exp(-rowmax) * lse, therefore rowmax + log(rowsum) cancels the effect of rowmax and outputs lse only.
    lse = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), float('inf'), rowmax_ * softmax_scale + tl.log(rowsum_))
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)
    
    # Rescale output
    if is_dropout:
        O_ *= inv_sum[:, None] * rpdrop
    else:
        O_ *= inv_sum[:, None]
    
    O = O_.to(O_ptr.type.element_ty)

    # Write back output
    O_ptr += bid * o_b_stride
    O_ptr += hid * o_h_stride
    O_offset = row_idx[:, None] * o_s_stride + tl.arange(0, HEAD_DIM)

    if IS_EVEN_MN:
        tl.store(O_ptr + O_offset, O)
    else:
        tl.store(O_ptr + O_offset, O, mask=qmask)
    
    # Write back lse
    lse_ptr += bid * hid * seqlen_q
    if IS_EVEN_MN:
        tl.store(lse_ptr + row_idx, lse)
    else:
        tl.store(lse_ptr + row_idx, lse, mask=row_idx < seqlen_q)


@triton.heuristics(
    values={
        'BLOCK_M': lambda args: block_m_splitkv_heuristic(args["HEAD_DIM"]),
        'BLOCK_N': lambda args: block_m_splitkv_heuristic(args["HEAD_DIM"]),
        'num_warps': lambda args: 4,
        'num_stages': lambda args: 3,
        'PRE_LOAD_V': lambda args: True,
        'IS_EVEN_MN': lambda args: (args["seqlen_q"] % args["BLOCK_M"] == 0) and (args["seqlen_k"] % args["BLOCK_N"] == 0),
    }
)
@triton.jit(do_not_specialize=["seqlen_q", "seqlen_k", "q_b_stride", "q_s_stride", "q_h_stride", "k_b_stride", "k_s_stride", "k_h_stride", "o_b_stride", "o_h_stride", "o_s_stride", "philox_seed", "philox_offset", "pdrop_u8", "slopes_batch_stride"])
def flash_fwd_splitkv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    P_ptr,
    O_ptr,
    lse_ptr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    q_b_stride,
    q_s_stride,
    q_h_stride,
    k_b_stride,
    k_s_stride,
    k_h_stride,
    o_b_stride,
    o_s_stride,
    o_h_stride,
    h,
    hk,
    pSlopes,
    philox_seed,
    philox_offset,
    pdrop_u8,
    rpdrop,
    slopes_batch_stride,
    HEAD_DIM: tl.constexpr,
    is_dropout: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    has_alibi: tl.constexpr,
    softmax_scale: tl.constexpr,
    softmax_scale_log2e: tl.constexpr,
    ws_left: tl.constexpr,
    ws_right: tl.constexpr,
    return_P: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_HEADS_K: tl.constexpr,
    IS_EVEN_MN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_splits: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr
):
    m_block = tl.program_id(0)
    split_id = tl.program_id(1)
    bid = tl.program_id(2) // NUM_HEADS
    hid = tl.program_id(2) % NUM_HEADS
    
    blocks_per_split = tl.cdiv(tl.cdiv(seqlen_k, BLOCK_N), num_splits)

    if is_local:
        n_block_min = max(split_id * blocks_per_split, (m_block * BLOCK_M + seqlen_k - seqlen_q - ws_left) / BLOCK_N)
    else:
        n_block_min = split_id * blocks_per_split

    n_block_max = min((split_id + 1) * blocks_per_split, tl.cdiv(seqlen_k, BLOCK_N))
    if is_causal or is_local:
        n_block_max = min(n_block_max,
                          tl.cdiv((m_block + 1) * BLOCK_M + seqlen_k - seqlen_q + ws_right, BLOCK_N))

    if has_alibi:
        alibi_offset = bid * slopes_batch_stride + hid
        alibi_slope = tl.load(pSlopes + alibi_offset)
        alibi_slope /= scale
    else:
        alibi_slope = 0.0

    if (not is_causal) and (not is_local):
        if IS_EVEN_MN:
            n_masking_blocks = 0
        else:
            n_masking_blocks = 1
    elif is_causal and IS_EVEN_MN: # causal implies ws_right is zero
        n_masking_blocks = tl.cdiv(BLOCK_M, BLOCK_N)
    else:
        # local and not causal, 
        n_masking_blocks = tl.cdiv(BLOCK_M, BLOCK_N) + 1

    Q_ptr += bid * q_b_stride
    Q_ptr += hid * q_h_stride
    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    Q_off = row_idx[:, None] * q_s_stride + tl.arange(0, HEAD_DIM)[None, :]
    qmask = row_idx[:, None] < seqlen_q
    if IS_EVEN_MN:
        Q = tl.load(Q_ptr + Q_off)
    else:
        Q = tl.load(Q_ptr + Q_off, mask=qmask)

    # Start from the right most block
    n_block = n_block_max - 1

    h_hk_ratio = h // hk
    K_ptr += bid * k_b_stride
    K_ptr += (hid // h_hk_ratio) * k_h_stride
    V_ptr += bid * k_b_stride
    V_ptr += (hid // h_hk_ratio) * k_h_stride
    
    P_ptr += ((bid * NUM_HEADS + hid) * seqlen_q_rounded + m_block * BLOCK_M) * seqlen_k_rounded
    P_ptr += n_block * BLOCK_N
    P_offset = tl.arange(0, BLOCK_M)[:, None] * seqlen_k_rounded + tl.arange(0, BLOCK_N)

    O_ = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    for n_block in tl.range(n_block_max - 1, n_block_max - n_masking_blocks - 1, step=-1):
        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        K_offset = col_idx[None, :] * k_s_stride + tl.arange(0, HEAD_DIM)[:, None]
        V_offset = col_idx[:, None] * k_s_stride + tl.arange(0, HEAD_DIM)[None, :]
        if IS_EVEN_MN:
            K = tl.load(K_ptr + K_offset, cache_modifier=".cg")
            if PRE_LOAD_V:
                V = tl.load(V_ptr + V_offset, cache_modifier=".cg")
        else:
            kvmask = col_idx < seqlen_k
            K = tl.load(K_ptr + K_offset, mask=kvmask[None, :], cache_modifier=".cg")
            if PRE_LOAD_V:
                V = tl.load(V_ptr + V_offset, mask=kvmask[:, None], cache_modifier=".cg")
        S = tl.dot(Q, K, allow_tf32=False)
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            seqlen_q,
            seqlen_k,
            ws_left,
            ws_right,
            alibi_slope,
            is_even_mn=IS_EVEN_MN,
            is_causal=is_causal,
            is_local=is_local,
            has_alibi=has_alibi
        )
        # col_idx -= BLOCK_N

        is_init = (n_block == n_block_max - 1).to(tl.int1)
        O_, P, rowmax_, rowsum_ = softmax_rescale(
            O_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=softmax_scale_log2e,
            is_border=(is_causal or is_local),
            is_init=is_init
        )
        P = P.to(Q_ptr.type.element_ty)

        if not PRE_LOAD_V:
            if IS_EVEN_MN:
                V = tl.load(V_ptr + V_offset, cache_modifier=".cg")
            else:
                V = tl.load(V_ptr + V_offset, mask=kvmask[:, None], cache_modifier=".cg")
        O_ = tl.dot(P, V, O_, allow_tf32=False)
        # if n_masking_blocks > 1 and n_block <= n_block_min:
        #     break

    for n_block in tl.range(n_block_max - n_masking_blocks - 1, n_block_min - 1, step=-1, num_stages=num_stages):
        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        K_offset = col_idx[None, :] * k_s_stride + tl.arange(0, HEAD_DIM)[:, None]
        K = tl.load(K_ptr + K_offset, cache_modifier=".cg")
        # if PRE_LOAD_V:
        #     V_offset = col_idx[:, None] * k_s_stride + tl.arange(0, HEAD_DIM)[None, :]
        #     V = tl.load(V_ptr + V_offset, cache_modifier=".cg")
        S = tl.dot(Q, K)

        if PRE_LOAD_V:
            V_offset = col_idx[:, None] * k_s_stride + tl.arange(0, HEAD_DIM)[None, :]
            V = tl.load(V_ptr + V_offset, cache_modifier=".cg")

        S = apply_mask(
            S,
            col_idx,
            row_idx,
            seqlen_q,
            seqlen_k,
            ws_left,
            ws_right,
            alibi_slope,
            is_even_mn=True,
            is_causal=False,
            is_local=is_local,
            has_alibi=has_alibi
        )
        # col_idx -= BLOCK_N

        is_init = (n_block == n_block_max - 1).to(tl.int1)
        O_, P, rowmax_, rowsum_ = softmax_rescale(
            O_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=softmax_scale_log2e,
            is_border=is_local,
            is_init=is_init
        )

        P = P.to(Q_ptr.type.element_ty)

        if not PRE_LOAD_V:
            V_offset = col_idx[:, None] * k_s_stride + tl.arange(0, HEAD_DIM)[None, :]
            V = tl.load(V_ptr + V_offset, cache_modifier=".cg")

        O_ = tl.dot(P, V, O_)

    # LSE
    lse = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), float('inf'), rowmax_ * softmax_scale + tl.log(rowsum_))
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)
    
    # Rescale output
    O_ *= inv_sum[:, None]

    # Write back output
    O_split_ptr = O_ptr
    # (n_splits, batch_size, num_heads, seqlen_q, head_size)
    # grid = (seq_block, split, batch * head)
    O_split_ptr += (split_id * tl.num_programs(2) + tl.program_id(2)) * seqlen_q * HEAD_DIM
    O_split_offset = row_idx[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)

    if IS_EVEN_MN:
        tl.store(O_split_ptr + O_split_offset, O_)
    else:
        tl.store(O_split_ptr + O_split_offset, O_, mask=qmask)
    
    # Write back lse
    lse_split_ptr = lse_ptr
    lse_split_ptr += (split_id * tl.num_programs(2) + tl.program_id(2)) * seqlen_q
    lse_split_ptr += m_block * BLOCK_M

    if IS_EVEN_MN:
        tl.store(lse_split_ptr + tl.arange(0, BLOCK_M), lse)
    else:
        tl.store(lse_split_ptr + tl.arange(0, BLOCK_M), lse, mask=row_idx < seqlen_q)


@triton.jit
def flash_fwd_splitkv_combine_kernel(
    out_ptr,
    lse_ptr,
    out_splits_ptr,
    lse_splits_ptr,
    head_size: tl.constexpr,
    out_b_stride,
    out_s_stride,
    out_h_stride,
    n_splits,
    BLOCK_M: tl.constexpr,
    q_total,
    MAX_N_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    lse_splits_ptr += pid * BLOCK_M
    lse_ptr += pid * BLOCK_M
    out_splits_ptr += pid * BLOCK_M * head_size
    out_ptr += pid * BLOCK_M * head_size
    lse_split_stride = tl.num_programs(0) * BLOCK_M
    out_split_stride = tl.num_programs(0) * BLOCK_M * head_size

    # Subtracting maximum from each of the split lse's for better numerical stability
    lse_split_offset = tl.arange(0, BLOCK_M)[:, None] + tl.arange(0, MAX_N_SPLITS)[None, :] * lse_split_stride
    lse_split_mask = (pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < q_total) & (tl.arange(0, MAX_N_SPLITS)[None, :] < n_splits)
    lse_splits = tl.load(lse_splits_ptr + lse_split_offset, mask=lse_split_mask, other=float('-inf'))
    max_lse = tl.max(lse_splits, 1)
    
    # Sum exp(lse(i) - max_lse) over all split i to obtain Z=sumexp(QK) up to a scaled factor exp(-max_lse)
    Zi_scaled = tl.exp(lse_splits - max_lse[:, None])
    Z_scaled = tl.sum(Zi_scaled, 1)
    Zi_Z = Zi_scaled / Z_scaled[:, None]

    # Write back LSE
    lse = tl.log(Z_scaled) + max_lse
    out_mask = pid * BLOCK_M + tl.arange(0, BLOCK_M) < q_total
    tl.store(lse_ptr + tl.arange(0, BLOCK_M), lse, mask=out_mask)

    out_split_offset = (
        tl.arange(0, BLOCK_M)[:, None, None] * head_size
        + tl.arange(0, MAX_N_SPLITS)[None, :, None] * out_split_stride
        + tl.arange(0, head_size)[None, None, :]
    )
    out_split_mask = (pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None, None] < q_total) & (tl.arange(0, MAX_N_SPLITS)[None, :, None] < n_splits)
    out_splits = tl.load(out_splits_ptr + out_split_offset, mask=out_split_mask, other=0)
    out = tl.sum(Zi_Z[:, :, None] * out_splits, 1)
    out = out.to(out_ptr.type.element_ty)
    
    # tl.device_print('O', out)
    # Write back output
    out_offset = tl.arange(0, BLOCK_M)[:, None] * out_s_stride + tl.arange(0, head_size)
    tl.store(out_ptr + out_offset, out, mask=out_mask[:, None])
    

def mha_fwd(
    q,
    k,
    v,
    out,
    alibi_slopes,
    p_dropout,
    softmax_scale,
    is_causal,
    window_size_left,
    window_size_right,
    return_softmax,
):
    q_dtype = q.dtype
    q_device = q.device
    assert q_dtype in (
        torch.float16,
        torch.bfloat16,
    ), "FlashAttention only support fp16 and bf16 data type"
    assert q_dtype == k.dtype
    assert q_dtype == v.dtype
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    batch_size, seqlen_q, num_heads, head_size = q.size()
    _, seqlen_k, num_heads_k, _ = k.size()
    assert (
        head_size % 8 == 0
    ), "head_size must be a multiple of 8, this is ensured by padding!"
    assert (
        num_heads % num_heads_k == 0
    ), "Number of heads in key/value must divide number of heads in query"
    if window_size_left >= seqlen_k:
        window_size_left = -1
    if window_size_right >= seqlen_k:
        window_size_right = -1
    if seqlen_q == 1 and alibi_slopes is None:
        is_causal = False
    if is_causal:
        window_size_right = 0

    if seqlen_q == 1 and num_heads > num_heads_k and window_size_left < 0 and window_size_right < 0 and p_dropout == 0 and not alibi_slopes:
        swap_seq_and_group = True
    else:
        swap_seq_and_group = False

    ngroups = num_heads // num_heads_k
    if swap_seq_and_group:
        q = q.reshape((batch_size, num_heads_k, ngroups, head_size)).transpose(1, 2)
        seqlen_q = ngroups
        num_heads = num_heads_k

    if out:
        assert out.stride(-1) == 1
        assert out.dtype == q.dtype
        assert out.size() == (batch_size, seqlen_q, num_heads, head_size)
    else:
        out = torch.empty_like(q, dtype=v.dtype)

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32)
    seqlen_q_rounded = round_multiple(seqlen_q, 128)
    seqlen_k_rounded = round_multiple(seqlen_k, 128)

    def splits_heuristics(num_tasks, num_sms, n_blocks):
        # splits only number of waves and wave efficiency are both low
        n_waves = triton.cdiv(num_tasks, num_sms)
        eff = (num_tasks / num_sms) / n_waves
        if eff > 0.85 or n_waves > 10:
            return 1
        max_eff = eff
        best_splits = 1
        for w in range(n_waves, 10):
            n_splits = min(num_sms, n_blocks, w * num_sms // num_tasks)
            blocks_per_split = triton.cdiv(n_blocks, n_splits)
            if blocks_per_split < 4:
                continue
            n_splits = triton.cdiv(n_blocks, blocks_per_split)
            eff = (n_splits * num_tasks / num_sms) / w
            if eff > max_eff:
                max_eff = eff
                best_splits = n_splits
        return best_splits

    with torch_device_fn.device(q_device):
        # Set softmax params
        lse = torch.empty((batch_size, num_heads, seqlen_q), dtype=torch.float, device=q_device)
        if return_softmax:
            assert p_dropout > 0, "return_softmax is only supported when p_dropout > 0.0"
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                dtype=q_dtype,
                device=q_device
            )
        else:
            p = torch.empty((), device=q_device)

        # Set dropout params
        if p_dropout > 0:
            increment = triton.cdiv(batch_size * num_heads * 32)
            philox_seed, philox_offset = update_philox_state(increment)
            is_dropout = True
        else:
            philox_seed, philox_offset = None, None
            is_dropout = False

        p_dropout = 1 - p_dropout
        pdrop_u8 = math.floor(p_dropout * 255.0)
        rpdrop = 1. / p_dropout

        # Check splitkv
        if not is_dropout:
            n_tasks = batch_size * num_heads * triton.cdiv(seqlen_q, block_m_splitkv_heuristic(head_size))
            num_sms = torch_device_fn.get_device_properties("cuda").multi_processor_count
            n_blocks = triton.cdiv(seqlen_k, block_n_splitkv_heuristic(head_size))
            n_splits = splits_heuristics(n_tasks, num_sms, n_blocks)
            print('n_blocks:', n_blocks)
            print('n_splits:', n_splits)
        else:
            n_splits = 1
        
        if n_splits > 1:
            lse_splits = torch.empty(
                (n_splits, batch_size, num_heads, seqlen_q),
                dtype=torch.float,
                device=q_device
            )
            out_splits = torch.empty(
                (n_splits, batch_size, num_heads, seqlen_q, head_size),
                dtype=torch.float,
                device=q_device
            )

        M_LOG2E	= 1.4426950408889634074
        softmax_scale_log2e = softmax_scale * M_LOG2E

        # Set alibi params
        if alibi_slopes is not None:
            assert alibi_slopes.device == q_device
            assert alibi_slopes.dtype in (torch.float, )
            assert alibi_slopes.stride(-1) == 1
            assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (batch_size, num_heads)
            alibi_slopes_batch_stride = alibi_slopes.stride(0) if alibi_slopes.ndim == 2 else 0
            has_alibi = True
        else:
            alibi_slopes_batch_stride = 0
            has_alibi = False

        # Set SWA params
        is_local = (window_size_left >= 0 or window_size_right >= 0) and not is_causal

        # ONLY EVEN_K IS SUPPORTED
        assert head_size == head_size_rounded

        # Launch kernel
        if n_splits > 1:
            grid = lambda args: (
                triton.cdiv(seqlen_q, args["BLOCK_M"]),
                n_splits,
                batch_size * num_heads
            )
            kernel = flash_fwd_splitkv_kernel[grid]
            tmp_lse = lse_splits
            tmp_out = out_splits
        else:
            grid = lambda args: (
                triton.cdiv(seqlen_q, args["BLOCK_M"]), # num_m_blocks
                batch_size,
                num_heads,
            )
            kernel = flash_fwd_kernel[grid]
            tmp_lse = lse
            tmp_out = out

        kernel = kernel(
            q,
            k,
            v,
            p,
            tmp_out,
            tmp_lse,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            seqlen_k_rounded,
            q.stride(0),
            q.stride(-3),
            q.stride(-2),
            k.stride(0),
            k.stride(-3),
            k.stride(-2),
            out.stride(0),
            out.stride(-3),
            out.stride(-2),
            num_heads,
            num_heads_k,
            alibi_slopes,
            philox_seed,
            philox_offset,
            pdrop_u8,
            rpdrop,
            alibi_slopes_batch_stride,
            head_size,
            is_dropout=is_dropout,
            is_causal=is_causal,
            is_local=is_local,
            has_alibi=has_alibi,
            softmax_scale=softmax_scale,
            softmax_scale_log2e=softmax_scale_log2e,
            ws_left=window_size_left,
            ws_right=window_size_right,
            return_P=return_softmax,
            BATCH_SIZE=batch_size,
            num_splits=n_splits,
            NUM_HEADS=num_heads,
            NUM_HEADS_K=num_heads_k,
        )
        print(f'{kernel.name} shared memory:', kernel.metadata.shared)
        # print(kernel.asm['ttgir'])
    
    if n_splits > 1:
        if head_size % 128 == 0:
            BLOCK_M = 4
        elif head_size % 64 == 0:
            BLOCK_M = 8
        else:
            BLOCK_M = 16
        grid = lambda args: (triton.cdiv(batch_size * num_heads * seqlen_q, BLOCK_M), )
        kernel = flash_fwd_splitkv_combine_kernel[grid](
            out,
            lse,
            tmp_out,
            tmp_lse,
            head_size,
            out.stride(0),
            out.stride(-3),
            out.stride(-1),
            n_splits,
            BLOCK_M,
            q_total=batch_size * num_heads * seqlen_q,
            MAX_N_SPLITS=triton.next_power_of_2(n_splits),
        )
        print(f'{kernel.name} shared memory:', kernel.metadata.shared)

    if swap_seq_and_group:
        out = out.transpose(1, 2).reshape((batch_size, 1, num_heads_k * seqlen_q, head_size))
        q = q.transpose(1, 2).reshape((batch_size, 1, num_heads_k * seqlen_q, head_size))
        lse = lse.reshape((batch_size, num_heads_k * seqlen_q, 1))

    return out, q, k, v, lse, philox_seed, philox_offset, p


def flash_attention_forward(
    query,
    key,
    value,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    return_debug_mask,
    *,
    scale=None,
    window_size_left=None,
    window_size_right=None,
    seqused_k=None,
    alibi_slopes=None
):
    logging.debug("GEMS FLASH_ATTENTION")
    assert cum_seq_q is None and cum_seq_k is None, "varlen is not supported yet."

    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    softmax_scale = scale or 1.0 / (HEAD_DIM_K**0.5)
    non_null_window_left = window_size_left or -1
    non_null_window_right = window_size_right or -1

    out, q, k, v, lse, philox_seed, philox_offset, p = mha_fwd(
        query,
        key,
        value,
        None,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        is_causal,
        non_null_window_left,
        non_null_window_right,
        return_debug_mask,
    )
    
    return (out, lse, philox_seed, philox_offset, p)
