import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import update_philox_state

from .. import runtime

logger = logging.getLogger(__name__)


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
    if HAS_ATTN_MASK:
        mask_block_ptr += lo * stride_attn_mask_kv_seqlen

    LOG2E: tl.constexpr = 1.44269504

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        kv_load_mask = (start_n + offs_n) < KV_CTX
        # start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr, mask=kv_load_mask[None, :], other=0.0)
        if PRE_LOAD_V:
            v = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)

        qk = tl.dot(q, k, allow_tf32=False)
        # incase not divisible.
        qk = tl.where(kv_load_mask[None, :], qk, -float("inf"))
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
            qk *= qk_scale * LOG2E
            if HAS_ATTN_MASK:
                qk = qk + attn_mask
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

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


def keep(cfg):
    BM = cfg.kwargs["BLOCK_M"]
    BN = cfg.kwargs["BLOCK_N"]
    w = cfg.num_warps

    return (BM, BN, w) in ((128, 32, 4), (128, 128, 8))


@triton.autotune(
    configs=list(filter(keep, runtime.get_tuned_config("attention"))),
    key=["KV_CTX", "HEAD_DIM"],
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
    o_offset = (
        batch_id.to(tl.int64) * stride_o_batch + head_id.to(tl.int64) * stride_o_head
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
        + o_offset
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
    logger.debug("GEMS SCALED DOT PRODUCT ATTENTION")
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
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.to(query.dtype) * -1.0e6
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


# Following implementation is largely a porting of TriDao's Flash Attention to Triton.
# Major difference can be found in dropout where the input to RNG is determined only
# by the element index in the attention score matrix. In contrast, the CUDA flash-attn
# employs a dropout that assumes an implementation specific threadblock data layout.


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
    c2, c3 = u64_to_lohi(subsequence.to(tl.uint64))

    # pragma unroll
    kPhiloxSA: tl.constexpr = 0xD2511F53
    kPhiloxSB: tl.constexpr = 0xCD9E8D57
    for _ in tl.static_range(6):
        res0 = kPhiloxSA * c0.to(tl.uint64)
        res1 = kPhiloxSB * c2.to(tl.uint64)
        res0_x, res0_y = u64_to_lohi(res0)
        res1_x, res1_y = u64_to_lohi(res1)
        c0, c1, c2, c3 = res1_y ^ c1 ^ k0, res1_x, res0_y ^ c3 ^ k1, res0_x
        k0 += kPhilox10A
        k1 += kPhilox10B

    res0 = kPhiloxSA * c0.to(tl.uint64)
    res1 = kPhiloxSB * c2.to(tl.uint64)
    res0_x, res0_y = u64_to_lohi(res0)
    res1_x, res1_y = u64_to_lohi(res1)
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
        P = tl.where(mask, P * 0, P)
    return P


@triton.jit
def apply_dropout(
    P,
    row_start,
    col_start,
    n_cols,
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
    row_start = tl.multiple_of(row_start, BLOCK_M)
    col_start = tl.multiple_of(col_start, BLOCK_N)
    row = row_start + tl.arange(0, BLOCK_M)[:, None]
    # Down scale col_idx by 4
    col = col_start // 4 + tl.arange(0, BLOCK_N // 4)[None, :]

    subsequence = row.to(tl.uint64) * n_cols + col.to(tl.uint64)

    offset = philox_offset + bid * NUM_HEADS + hid
    offset += subsequence * 0
    r0, r1, r2, r3 = philox_(philox_seed, subsequence, offset)

    r = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(BLOCK_M, BLOCK_N)

    mask = (r & 0xFF) >= p_dropout_uint8

    P = apply_dropout_mask(
        P, mask, encode_dropout_in_sign_bit=encode_dropout_in_sign_bit
    )
    return P


@triton.jit
def apply_mask(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    ws_left,
    ws_right,
    is_even_mn: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    has_alibi: tl.constexpr,
    alibi_slope: tl.constexpr = None,
):
    need_mask: tl.constexpr = is_causal | has_alibi | is_local | (not is_even_mn)
    if need_mask:
        # Extra care should be taken to void one-off errors: both col_lb and col_rb are inclusive!
        col_lb = max(0, row_idx + max_seqlen_k - max_seqlen_q - ws_left)
        col_rb = min(max_seqlen_k - 1, row_idx + max_seqlen_k - max_seqlen_q + ws_right)

        if has_alibi:
            S -= alibi_slope * tl.abs(col_idx[None, :] - row_idx[:, None])

        if is_causal:
            S = tl.where(col_idx[None, :] > col_rb[:, None], float("-inf"), S)

        if is_local:
            S = tl.where(
                (col_idx[None, :] > col_rb[:, None])
                | (col_idx[None, :] < col_lb[:, None]),
                float("-inf"),
                S,
            )

        if (not is_local) & (not is_causal) & (not is_even_mn):
            S = tl.where(col_idx[None, :] >= max_seqlen_k, float("-inf"), S)

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

    if is_border:
        cur_max = tl.where(row_max == float("-inf"), 0, row_max)
    else:
        cur_max = row_max

    p_scale = tl.math.exp2((prev_max - cur_max) * softmax_scale_log2e)
    row_sum *= p_scale
    O_acc *= p_scale[:, None]

    max_scaled = tl.where(row_max == float("-inf"), 0, row_max * softmax_scale_log2e)
    P = tl.math.exp2(S * softmax_scale_log2e - max_scaled[:, None])
    row_sum = row_sum + tl.sum(P, 1)
    return O_acc, P, row_max, row_sum


def block_m_splitkv_heuristic(headdim):
    return 128 if headdim <= 128 else 64


def block_n_splitkv_heuristic(headdim):
    return 64 if headdim <= 64 else 32


def is_even_mn(M, N, BM, BN, WL, WR):
    if M % BM == 0 and N % BN == 0:
        if M % N == 0 or N % M == 0:
            if (WL == -1 or WL % BN == 0) and (WR == -1 or WR % BN == 0):
                return True
    return False


@triton.autotune(
    configs=list(filter(keep, runtime.get_tuned_config("attention"))),
    key=["HEAD_DIM"],
)
@triton.heuristics(
    values={
        "PRE_LOAD_V": lambda args: False,
        "IS_EVEN_MN": lambda args: is_even_mn(
            args["seqlen_q"],
            args["seqlen_k"],
            args["BLOCK_M"],
            args["BLOCK_N"],
            args["ws_left"],
            args["ws_right"],
        ),
    }
)
@triton.jit(do_not_specialize=["seqlen_q", "seqlen_k", "philox_seed", "philox_offset"])
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
    h: tl.constexpr,
    hk: tl.constexpr,
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
    blocks_per_split: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    m_block = tl.program_id(0)
    bh = tl.program_id(1)
    hid = bh % h
    bid = bh // h
    num_m_blocks = tl.cdiv(seqlen_q, BLOCK_M)

    # We draw a minimum covering frame on the attention map that this CTA is assigned to process.
    # The frame edges are rounded to multiples of BLOCK_M and BLOCK_N for rows and columns respectively.

    col_min = 0
    if is_local:
        col_min = max(0, m_block * BLOCK_M + seqlen_k - seqlen_q - ws_left)
        if not IS_EVEN_MN:
            # round left
            col_min = (col_min // BLOCK_N) * BLOCK_N

    col_max = seqlen_k
    if is_causal or is_local:
        col_max += (m_block - num_m_blocks + 1) * BLOCK_M
        if is_local:
            col_max += ws_right
        col_max = min(seqlen_k, col_max)

    if not IS_EVEN_MN:
        # round right
        col_max = tl.cdiv(col_max, BLOCK_N) * BLOCK_N

    if (not is_causal) and (not is_local):
        if IS_EVEN_MN:
            masking_cols: tl.constexpr = 0
        else:
            masking_cols: tl.constexpr = BLOCK_N
    elif (is_causal | is_local) and IS_EVEN_MN:  # causal implies ws_right is zero
        masking_cols: tl.constexpr = tl.cdiv(BLOCK_M, BLOCK_N) * BLOCK_N
    else:
        # local
        masking_cols: tl.constexpr = (tl.cdiv(BLOCK_M, BLOCK_N) + 1) * BLOCK_N

    if is_dropout:
        philox_seed = tl.load(philox_seed).to(tl.uint64)
        philox_offset = tl.load(philox_offset).to(tl.uint64)

    if has_alibi:
        alibi_offset = bid * slopes_batch_stride + hid
        alibi_slope = tl.load(pSlopes + alibi_offset)
        alibi_slope /= softmax_scale
    else:
        alibi_slope = 0.0

    q_b_stride = tl.multiple_of(q_b_stride, HEAD_DIM * h)
    Q_ptr += bid * q_b_stride
    Q_ptr += hid * q_h_stride
    row_start = m_block * BLOCK_M
    row_idx = row_start + tl.arange(0, BLOCK_M)
    Q_off = row_idx[:, None] * q_s_stride + tl.arange(0, HEAD_DIM)[None, :]
    qmask = row_idx[:, None] < seqlen_q
    if IS_EVEN_MN:
        Q = tl.load(Q_ptr + Q_off, cache_modifier=".cg")
    else:
        Q = tl.load(Q_ptr + Q_off, mask=qmask, cache_modifier=".cg")

    if return_P:
        P_ptr += (
            (bid * NUM_HEADS + hid) * seqlen_q_rounded + m_block * BLOCK_M
        ) * seqlen_k_rounded
        P_offset = tl.arange(0, BLOCK_M)[:, None] * seqlen_k_rounded + tl.arange(
            0, BLOCK_N
        )
        p_bp0 = P_ptr + P_offset

    O_ = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    k_b_stride = tl.multiple_of(k_b_stride, HEAD_DIM * hk)
    h_hk_ratio = h // hk
    K_ptr += bid * k_b_stride
    K_ptr += (hid // h_hk_ratio) * k_h_stride
    V_ptr += bid * k_b_stride
    V_ptr += (hid // h_hk_ratio) * k_h_stride

    K_offset = (
        tl.arange(0, BLOCK_N)[None, :] * k_s_stride + tl.arange(0, HEAD_DIM)[:, None]
    )
    V_offset = (
        tl.arange(0, BLOCK_N)[:, None] * k_s_stride + tl.arange(0, HEAD_DIM)[None, :]
    )

    p_bk0 = K_ptr + K_offset
    p_bv0 = V_ptr + V_offset

    if is_causal | is_local | (not IS_EVEN_MN):
        # Cut short masking cols if there's not enough cols out there
        masking_cols = min(col_max - col_min, masking_cols)
        for col_shift in tl.range(0, masking_cols, step=BLOCK_N):
            col_start = col_max - col_shift - BLOCK_N
            col_start = tl.multiple_of(col_start, BLOCK_N)
            off = col_start * k_s_stride
            if IS_EVEN_MN:
                K = tl.load(p_bk0 + off, cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_bv0 + off, cache_modifier=".cg")
            else:
                col_idx = col_start + tl.arange(0, BLOCK_N)
                kvmask = col_idx < seqlen_k
                K = tl.load(p_bk0 + off, mask=kvmask[None, :], cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_bv0 + off, mask=kvmask[:, None], cache_modifier=".cg")
            S = tl.dot(Q, K, allow_tf32=False)
            col_idx = col_start + tl.arange(0, BLOCK_N)
            row_idx = row_start + tl.arange(0, BLOCK_M)

            # tl.store(p_bp0 + col_start, S)
            S = apply_mask(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                ws_left,
                ws_right,
                is_even_mn=IS_EVEN_MN,
                is_causal=is_causal,
                is_local=is_local,
                has_alibi=has_alibi,
                alibi_slope=alibi_slope,
            )

            O_, P, rowmax_, rowsum_ = softmax_rescale(
                O_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=softmax_scale_log2e,
                is_border=(is_causal or is_local),
            )
            P = P.to(V_ptr.type.element_ty)

            if is_dropout:
                if return_P:
                    P_drop = P

                    P_drop = apply_dropout(
                        P_drop,
                        row_start,
                        col_start,
                        seqlen_k,
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
                    if IS_EVEN_MN:
                        tl.store(p_bp0 + col_start, P_drop)
                    else:
                        kvmask = col_idx < seqlen_k
                        tl.store(
                            p_bp0 + col_start, P_drop, mask=qmask & kvmask[None, :]
                        )

                P = apply_dropout(
                    P,
                    row_start,
                    col_start,
                    seqlen_k,
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
                    kvmask = col_idx < seqlen_k
                    V = tl.load(p_bv0 + off, mask=kvmask[:, None], cache_modifier=".cg")
            O_ = tl.dot(P, V, O_, allow_tf32=False)

    for col_start in tl.range(
        col_min, col_max - masking_cols, step=BLOCK_N, num_stages=num_stages
    ):
        col_start = tl.multiple_of(col_start, BLOCK_N)
        off = col_start * k_s_stride
        K = tl.load(p_bk0 + off, cache_modifier=".cg")
        if PRE_LOAD_V:
            V = tl.load(p_bv0 + off, cache_modifier=".cg")
        S = tl.dot(Q, K)

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
            is_even_mn=True,
            is_causal=False,
            is_local=is_local,
            has_alibi=has_alibi,
            alibi_slope=alibi_slope,
        )

        O_, P, rowmax_, rowsum_ = softmax_rescale(
            O_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=softmax_scale_log2e,
            is_border=is_local,
        )
        P = P.to(V_ptr.type.element_ty)

        if is_dropout:
            if return_P:
                P_drop = P
                P_drop = apply_dropout(
                    P_drop,
                    row_start,
                    col_start,
                    seqlen_k,
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
                if IS_EVEN_MN:
                    tl.store(p_bp0 + col_start, P_drop)
                else:
                    kvmask = col_idx < seqlen_k
                    tl.store(p_bp0 + col_start, P_drop, mask=qmask & kvmask[None, :])

            P = apply_dropout(
                P,
                row_start,
                col_start,
                seqlen_k,
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
    # Note, rowsum = exp(-rowmax) * exp(lse), therefore rowmax + log(rowsum) cancels
    # the effect of rowmax and outputs lse only.
    lse = tl.where(
        rowsum_ == 0 | (rowsum_ != rowsum_),
        float("inf"),
        rowmax_ * softmax_scale + tl.log(rowsum_),
    )
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)

    if is_dropout:
        O_ *= inv_sum[:, None] * rpdrop
    else:
        O_ *= inv_sum[:, None]

    O = O_.to(O_ptr.type.element_ty)  # noqa

    # Write back output
    o_b_stride = tl.multiple_of(o_b_stride, HEAD_DIM * h)
    O_ptr += bid * o_b_stride
    O_ptr += hid * o_h_stride
    O_offset = row_idx[:, None] * o_s_stride + tl.arange(0, HEAD_DIM)

    if IS_EVEN_MN:
        tl.store(O_ptr + O_offset, O)
    else:
        tl.store(O_ptr + O_offset, O, mask=qmask)

    # Write back lse
    p_lse = lse_ptr + (bid * h + hid) * seqlen_q
    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)

    if IS_EVEN_MN:
        tl.store(p_lse + row_idx, lse)
    else:
        tl.store(p_lse + row_idx, lse, mask=row_idx < seqlen_q)


@triton.autotune(
    configs=list(filter(keep, runtime.get_tuned_config("attention"))),
    key=["HEAD_DIM"],
)
@triton.heuristics(
    values={
        "PRE_LOAD_V": lambda args: True,
        "IS_EVEN_MN": lambda args: is_even_mn(
            args["seqlen_q"],
            args["seqlen_k"],
            args["BLOCK_M"],
            args["BLOCK_N"],
            args["ws_left"],
            args["ws_right"],
        ),
    }
)
@triton.jit(do_not_specialize=["seqlen_q", "seqlen_k", "philox_seed", "philox_offset"])
def flash_fwd_bh_parallel_kernel(
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
    blocks_per_split: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    # (TODO)
    pass


# @triton.autotune(
#     configs=list(filter(keep, runtime.get_tuned_config("attention"))),
#     key=["HEAD_DIM"],
# )
@triton.heuristics(
    values={
        "BLOCK_M": lambda args: block_m_splitkv_heuristic(args["HEAD_DIM"]),
        "BLOCK_N": lambda args: block_n_splitkv_heuristic(args["HEAD_DIM"]),
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 3,
        "PRE_LOAD_V": lambda args: True,
        "IS_EVEN_MN": lambda args: is_even_mn(
            args["seqlen_q"],
            args["seqlen_k"],
            args["BLOCK_M"],
            args["BLOCK_N"],
            args["ws_left"],
            args["ws_right"],
        ),
    }
)
@triton.jit(do_not_specialize=["seqlen_q", "seqlen_k", "philox_seed", "philox_offset"])
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
    blocks_per_split: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    m_block = tl.program_id(0)
    split_id = tl.program_id(1)
    bid = tl.program_id(2) // NUM_HEADS
    hid = tl.program_id(2) % NUM_HEADS

    split_block_min = split_id * blocks_per_split
    split_block_max = split_block_min + blocks_per_split

    n_block_max = tl.cdiv(seqlen_k, BLOCK_N)
    if is_causal:
        n_block_max = min(
            n_block_max,
            tl.cdiv((m_block + 1) * BLOCK_M + seqlen_k - seqlen_q + ws_right, BLOCK_N),
        )

    if has_alibi:
        alibi_offset = bid * slopes_batch_stride + hid
        alibi_slope = tl.load(pSlopes + alibi_offset)
        alibi_slope /= softmax_scale
    else:
        alibi_slope = 0

    if not is_causal:
        if IS_EVEN_MN:
            masking_block_min = n_block_max
        else:
            masking_block_min = n_block_max - 1
    elif is_causal and IS_EVEN_MN:  # causal implies ws_right is zero
        masking_block_min = n_block_max - tl.cdiv(BLOCK_M, BLOCK_N)
    else:
        masking_block_min = n_block_max - tl.cdiv(BLOCK_M, BLOCK_N) - 1

    Q_ptr += bid * q_b_stride
    Q_ptr += hid * q_h_stride
    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    Q_off = row_idx[:, None] * q_s_stride + tl.arange(0, HEAD_DIM)[None, :]
    p_qm = Q_ptr + Q_off
    qmask = row_idx[:, None] < seqlen_q
    if IS_EVEN_MN:
        Q = tl.load(p_qm, cache_modifier=".cg")
    else:
        Q = tl.load(p_qm, mask=qmask, cache_modifier=".cg")

    h_hk_ratio = h // hk
    K_ptr += bid * k_b_stride
    K_ptr += (hid // h_hk_ratio) * k_h_stride
    V_ptr += bid * k_b_stride
    V_ptr += (hid // h_hk_ratio) * k_h_stride

    K_offset = (
        tl.arange(0, BLOCK_N)[None, :] * k_s_stride + tl.arange(0, HEAD_DIM)[:, None]
    )
    p_k0 = K_ptr + K_offset

    V_offset = (
        tl.arange(0, BLOCK_N)[:, None] * k_s_stride + tl.arange(0, HEAD_DIM)[None, :]
    )
    p_v0 = V_ptr + V_offset

    O_ = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    if split_block_max <= masking_block_min:
        # no masking needed
        for n_block in tl.range(
            split_block_min, split_block_max, num_stages=num_stages
        ):
            kv_off = n_block * BLOCK_N * k_s_stride
            K = tl.load(p_k0 + kv_off, cache_modifier=".cg")
            if PRE_LOAD_V:
                V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
            S = tl.dot(Q, K)

            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

            if has_alibi:
                S -= alibi_slope * tl.abs(col_idx[None, :] - row_idx[:, None])

            O_, P, rowmax_, rowsum_ = softmax_rescale(
                O_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=softmax_scale_log2e,
                is_border=False,
            )

            if not PRE_LOAD_V:
                V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
            P = P.to(Q_ptr.type.element_ty)
            O_ = tl.dot(P, V, O_)
    else:
        for n_block in tl.range(split_block_min, min(split_block_max, n_block_max)):
            kv_off = n_block * BLOCK_N * k_s_stride
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
            if IS_EVEN_MN:
                K = tl.load(p_k0 + kv_off, cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
            else:
                kvmask = col_idx < seqlen_k
                K = tl.load(p_k0 + kv_off, mask=kvmask[None, :], cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(
                        p_v0 + kv_off, mask=kvmask[:, None], cache_modifier=".cg"
                    )

            S = tl.dot(Q, K)

            S = apply_mask(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                ws_left,
                ws_right,
                is_even_mn=IS_EVEN_MN,
                is_causal=is_causal,
                is_local=False,
                has_alibi=has_alibi,
                alibi_slope=alibi_slope,
            )

            O_, P, rowmax_, rowsum_ = softmax_rescale(
                O_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=softmax_scale_log2e,
                is_border=(is_causal or is_local),
            )

            if not PRE_LOAD_V:
                if IS_EVEN_MN:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
                else:
                    V = tl.load(
                        p_v0 + kv_off, mask=kvmask[:, None], cache_modifier=".cg"
                    )
            P = P.to(Q_ptr.type.element_ty)
            O_ = tl.dot(P, V, O_)

    # LSE
    lse = tl.where(
        rowsum_ == 0 | (rowsum_ != rowsum_),
        float("-inf"),
        rowmax_ * softmax_scale + tl.log(rowsum_),
    )
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)

    # Rescale output
    O_ *= inv_sum[:, None]

    # Write back output
    # O_splits layout = (n_splits, batch_size, num_heads, seqlen_q, head_size)
    # grid = (seq_block, split, batch * head)
    O_split_ptr = O_ptr
    # + split, batch, head offsets, seq_block offsets are already added in row_idx
    O_split_ptr += (
        (split_id * tl.num_programs(2) + tl.program_id(2)) * seqlen_q * HEAD_DIM
    )
    O_split_offset = row_idx[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)
    O_split_ptr = tl.multiple_of(O_split_ptr, HEAD_DIM)
    p_om = O_split_ptr + O_split_offset

    if IS_EVEN_MN:
        tl.store(p_om, O_, cache_modifier=".cg")
    else:
        tl.store(p_om, O_, mask=qmask, cache_modifier=".cg")

    # Write back lse
    # lse_splits layout = (n_splits, batch_size, num_heads, seqlen_q)
    lse_split_ptr = lse_ptr
    # + split, batch, head, seq_block offsets
    lse_split_ptr += (
        split_id * tl.num_programs(2) + tl.program_id(2)
    ) * seqlen_q + m_block * BLOCK_M

    if IS_EVEN_MN:
        tl.store(lse_split_ptr + tl.arange(0, BLOCK_M), lse, cache_modifier=".cg")
    else:
        tl.store(
            lse_split_ptr + tl.arange(0, BLOCK_M),
            lse,
            mask=row_idx < seqlen_q,
            cache_modifier=".cg",
        )


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
    lse_split_offset = (
        tl.arange(0, BLOCK_M)[:, None]
        + tl.arange(0, MAX_N_SPLITS)[None, :] * lse_split_stride
    )
    lse_split_mask = (pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < q_total) & (
        tl.arange(0, MAX_N_SPLITS)[None, :] < n_splits
    )
    lse_splits = tl.load(
        lse_splits_ptr + lse_split_offset, mask=lse_split_mask, other=float("-inf")
    )
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
    out_split_mask = (
        pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None, None] < q_total
    ) & (tl.arange(0, MAX_N_SPLITS)[None, :, None] < n_splits)
    out_splits = tl.load(
        out_splits_ptr + out_split_offset, mask=out_split_mask, other=0
    )
    out = tl.sum(Zi_Z[:, :, None] * out_splits, 1)
    out = out.to(out_ptr.type.element_ty)

    # Write back output
    out_offset = tl.arange(0, BLOCK_M)[:, None] * out_s_stride + tl.arange(0, head_size)
    tl.store(out_ptr + out_offset, out, mask=out_mask[:, None])


_debug = False


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
    disable_splitkv=False,
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

    if (
        seqlen_q == 1
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
        and p_dropout == 0
        and not alibi_slopes
    ):
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
    seqlen_k_rounded = round_multiple(seqlen_k, 32)

    def splits_heuristic(num_tasks, num_sms, n_blocks):
        # splits when wave efficiency is low
        n_waves = triton.cdiv(num_tasks, num_sms)
        eff = (num_tasks / num_sms) / n_waves
        if eff > 0.8 or n_waves > 1:
            return 1

        min_blocks_per_split = 2
        best_splits = min(
            triton.cdiv(n_blocks, min_blocks_per_split),
            int(math.floor(1.0 / eff)),
            num_sms,
        )

        # best_splits = 1
        # best_eff = eff
        # min_blocks_per_split = 1
        # max_blocks_per_split = triton.cdiv(n_blocks, 2)
        # for blocks_per_split in range(min_blocks_per_split, max_blocks_per_split + 1)[::-1]:
        #     n_splits = triton.cdiv(n_blocks, blocks_per_split)
        #     n_waves = triton.cdiv(n_splits * num_tasks, num_sms)
        #     eff = (n_splits * num_tasks / num_sms) / n_waves
        #     if eff > 0.85:
        #         best_splits = n_splits
        #         break
        return best_splits

    with torch_device_fn.device(q_device):
        # Set softmax params
        lse = torch.empty(
            (batch_size, num_heads, seqlen_q), dtype=torch.float, device=q_device
        )
        if return_softmax:
            assert (
                p_dropout > 0
            ), "return_softmax is only supported when p_dropout > 0.0"
            p = torch.zeros(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                dtype=q_dtype,
                device=q_device,
            )
        else:
            p = torch.empty((), device=q_device)

        # Set dropout params
        if p_dropout > 0:
            increment = batch_size * num_heads * 32
            philox_seed, philox_offset = update_philox_state(increment)
            philox_seed = torch.tensor(philox_seed, dtype=torch.int64, device=q_device)
            philox_offset = torch.tensor(
                philox_offset, dtype=torch.int64, device=q_device
            )
            is_dropout = True
        else:
            philox_seed, philox_offset = None, None
            is_dropout = False

        p_dropout = 1 - p_dropout
        pdrop_u8 = math.floor(p_dropout * 255.0)
        rpdrop = 1.0 / p_dropout

        M_LOG2E = 1.4426950408889634074
        softmax_scale_log2e = softmax_scale * M_LOG2E

        # Set alibi params
        if alibi_slopes is not None:
            assert alibi_slopes.device == q_device
            assert alibi_slopes.dtype in (torch.float,)
            assert alibi_slopes.stride(-1) == 1
            assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (
                batch_size,
                num_heads,
            )
            alibi_slopes_batch_stride = (
                alibi_slopes.stride(0) if alibi_slopes.ndim == 2 else 0
            )
            has_alibi = True
        else:
            alibi_slopes_batch_stride = 0
            has_alibi = False

        # Set SWA params
        is_causal = window_size_left < 0 and window_size_right == 0
        is_local = window_size_left >= 0 and window_size_right >= 0

        # ONLY EVEN_K IS SUPPORTED
        assert head_size == head_size_rounded

        # Do kernel dispatching
        def dispatch(B, H, Q, K, D):
            num_sms = torch_device_fn.get_device_properties(
                "cuda"
            ).multi_processor_count

            default_args = {}

            # Try bh parallel
            # if B * H > 0.8 * num_sms:
            #     kernel = flash_fwd_bh_parallel_kernel[(H, B)]
            #     # Yield kernel and prefilled args
            #     return kernel, default_args, None, None

            # Try splitkv
            if not is_dropout and not is_local and not disable_splitkv:
                BM = block_m_splitkv_heuristic(D)
                n_tasks = B * H * triton.cdiv(seqlen_q, BM)
                BN = block_n_splitkv_heuristic(D)
                n_blocks = triton.cdiv(seqlen_k, BN)
                n_splits = splits_heuristic(n_tasks, num_sms, n_blocks)

                if _debug:
                    n_splits = 32
                    n_blocks = triton.cdiv(K, BN)
                    blocks_per_split = triton.cdiv(n_blocks, n_splits)
                    print("block_n:", BN)
                    print("n_splits:", n_splits)
                    print("blocks_per_split", blocks_per_split)

                if n_splits > 1:
                    lse_splits = torch.empty(
                        (n_splits, B, H, Q), dtype=torch.float, device=q_device
                    )
                    out_splits = torch.empty(
                        (n_splits, B, H, Q, D), dtype=torch.float, device=q_device
                    )
                    grid = lambda args: (
                        triton.cdiv(Q, args["BLOCK_M"]),
                        n_splits,
                        B * H,
                    )
                    splitkv_kernel = flash_fwd_splitkv_kernel[grid]
                    blocks_per_split = triton.cdiv(n_blocks, n_splits)
                    splitkv_args = default_args.copy()
                    splitkv_args["blocks_per_split"] = blocks_per_split
                    splitkv_args["O_ptr"] = out_splits
                    splitkv_args["lse_ptr"] = lse_splits
                    # kernel = yield kernel, args

                    if D % 128 == 0:
                        BLOCK_M = 4
                    elif D % 64 == 0:
                        BLOCK_M = 8
                    else:
                        BLOCK_M = 16
                    grid = lambda args: (triton.cdiv(B * H * Q, BLOCK_M),)
                    combine_kernel = flash_fwd_splitkv_combine_kernel[grid]
                    combine_args = {
                        "out_splits_ptr": out_splits,
                        "lse_splits_ptr": lse_splits,
                        "n_splits": n_splits,
                        "BLOCK_M": BLOCK_M,
                        "q_total": B * H * Q,
                        "MAX_N_SPLITS": triton.next_power_of_2(n_splits),
                    }
                    return splitkv_kernel, splitkv_args, combine_kernel, combine_args

            # Last option: flash_fwd
            grid = lambda args: (
                triton.cdiv(Q, args["BLOCK_M"]),
                H * B,
            )
            kernel = flash_fwd_kernel[grid]
            return kernel, default_args, None, None

        kernel1, kernel1_args, kernel2, kernel2_args = dispatch(
            batch_size, num_heads, seqlen_q, seqlen_k, head_size
        )

        if _debug:
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                dtype=torch.float32,
                device=q_device,
            )
            return_softmax = True

        prefilled_args = {
            "Q_ptr": q,
            "K_ptr": k,
            "V_ptr": v,
            "P_ptr": p,
            "O_ptr": out,
            "lse_ptr": lse,
            "seqlen_q": seqlen_q,
            "seqlen_k": seqlen_k,
            "seqlen_q_rounded": seqlen_q_rounded,
            "seqlen_k_rounded": seqlen_k_rounded,
            "q_b_stride": q.stride(0),
            "q_s_stride": q.stride(-3),
            "q_h_stride": q.stride(-2),
            "k_b_stride": k.stride(0),
            "k_s_stride": k.stride(-3),
            "k_h_stride": k.stride(-2),
            "o_b_stride": out.stride(0),
            "o_s_stride": out.stride(-3),
            "o_h_stride": out.stride(-2),
            "h": num_heads,
            "hk": num_heads_k,
            "pSlopes": alibi_slopes,
            "philox_seed": philox_seed,
            "philox_offset": philox_offset,
            "pdrop_u8": pdrop_u8,
            "rpdrop": rpdrop,
            "slopes_batch_stride": alibi_slopes_batch_stride,
            "HEAD_DIM": head_size,
            "is_dropout": is_dropout,
            "is_causal": is_causal,
            "is_local": is_local,
            "has_alibi": has_alibi,
            "softmax_scale": softmax_scale,
            "softmax_scale_log2e": softmax_scale_log2e,
            "ws_left": window_size_left,
            "ws_right": window_size_right,
            "return_P": return_softmax,
            "BATCH_SIZE": batch_size,
            "blocks_per_split": None,
            "NUM_HEADS": num_heads,
            "NUM_HEADS_K": num_heads_k,
        }

        args_copy = prefilled_args.copy()
        args_copy.update(kernel1_args)

        kernel = kernel1(**args_copy)
        if _debug:
            print(f"{kernel.name} shared memory:", kernel.metadata.shared)
            print(f"{kernel.name} num_warps:", kernel.metadata.num_warps)
            print(f"{kernel.name} num_stages:", kernel.metadata.num_stages)
            # print(kernel.asm['ttgir'])
            print("p:", p)

        # Combine
        if kernel2 is not None:
            prefilled_args = {
                "out_ptr": out,
                "lse_ptr": lse,
                "head_size": head_size,
                "out_b_stride": out.stride(0),
                "out_s_stride": out.stride(-3),
                "out_h_stride": out.stride(-1),
            }
            args_copy = prefilled_args.copy()
            args_copy.update(kernel2_args)
            kernel2(**args_copy)

    if swap_seq_and_group:
        out = out.transpose(1, 2).reshape(
            (batch_size, 1, num_heads_k * seqlen_q, head_size)
        )
        q = q.transpose(1, 2).reshape(
            (batch_size, 1, num_heads_k * seqlen_q, head_size)
        )
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
    alibi_slopes=None,
    disable_splitkv=False,
):
    logging.debug("GEMS FLASH_ATTENTION_FORWARD")
    assert cum_seq_q is None and cum_seq_k is None, "varlen is not supported yet."

    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    softmax_scale = scale or 1.0 / (HEAD_DIM_K**0.5)
    if window_size_left is not None:
        non_null_window_left = window_size_left
    else:
        non_null_window_left = -1
    if window_size_right is not None:
        non_null_window_right = window_size_right
    else:
        non_null_window_right = -1

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
        disable_splitkv=disable_splitkv,
    )

    return (out, lse, philox_seed, philox_offset, p)
