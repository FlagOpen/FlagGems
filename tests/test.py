import logging
import math 

import torch
import triton
import triton.language as tl

from loguru import logger 


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    query,  #
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

    # loop over key, value and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        kv_load_mask = (start_n + offs_n) < KV_CTX
        # start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        key = tl.load(K_block_ptr, mask=kv_load_mask[None, :], other=0.0)
        if PRE_LOAD_V:
            value = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)

        qk = tl.dot(query, key, allow_tf32=False)
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
            value = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(query.dtype)
        p = p.to(value.dtype)
        acc = tl.dot(p, value, acc, allow_tf32=False)
        # update m_i and l_i
        m_i = m_ij

        K_block_ptr += BLOCK_N * stride_k_seqlen
        V_block_ptr += BLOCK_N * stride_v_seqlen

        if HAS_ATTN_MASK:
            mask_block_ptr += BLOCK_N * stride_attn_mask_kv_seqlen

    return acc, l_i, m_i


def early_config_prune(configs, nargs, **kwargs):
    return list(filter(lambda cfg: cfg.kwargs["BLOCK_N"] <= nargs["HEAD_DIM"], configs))


configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN, "PRE_LOAD_V": 0}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64]
    for s in [2, 3, 4]
    for w in [4, 8]
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(
    # configs=runtime.get_tuned_config("attention"),
    configs=list(filter(keep, configs)),
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
    M, 
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
    kv_head_num,
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
    kv_head_id = off_hz % kv_head_num

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
    # load query: it will stay in SRAM throughout
    query = tl.load(Q_block_ptr, mask=q_load_mask[:, None], other=0.0)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            query,
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
            query,
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
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * Q_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=q_load_mask[:, None])


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = off_m < N_CTX

    off_hz = tl.program_id(1)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(O + off_hz * D_HEAD * N_CTX + off_m[:, None] * D_HEAD + off_n[None, :], mask=mask[:, None], other=0.0)
    do = tl.load(DO + off_hz * D_HEAD * N_CTX + off_m[:, None] * D_HEAD + off_n[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta, mask=off_m < N_CTX)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, key, value, sm_scale,  #
                   DO,  #
                   M, D,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   BLOCK_DMODEL: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_m_mask = offs_m < N_CTX

    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_n_mask = offs_n < N_CTX

    offs_k = tl.arange(0, BLOCK_DMODEL)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        # qT = tl.load(qT_ptrs)
        qT = tl.load(qT_ptrs, mask=offs_m_mask[None, :], other=0.0)

        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m, mask=offs_m_mask, other=float("inf"))


        qkT = tl.dot(key, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None]) & (offs_m < N_CTX)[None, :] & (offs_n < N_CTX)[:, None]
            pT = tl.where(mask, pT, 0.0)


        # tl.device_print("pT: ", pT)
        # do = tl.load(do_ptrs)
        do = tl.load(do_ptrs, mask=offs_m_mask[:, None], other=0.0)

        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        # Di = tl.load(D + offs_m)
        Di = tl.load(D + offs_m, mask=offs_m_mask, other=0.0)

        # Compute dP and dS.
        dpT = tl.dot(value, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, query, K, V,  #
                 do, m, D,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 BLOCK_DMODEL: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_m_mask = offs_m < N_CTX

    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_n_mask = offs_n < N_CTX

    offs_k = tl.arange(0, BLOCK_DMODEL)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        # kT = tl.load(kT_ptrs)
        # vT = tl.load(vT_ptrs)

        kT = tl.load(kT_ptrs, mask=offs_n_mask[None, :], other=0.0)
        vT = tl.load(vT_ptrs, mask=offs_n_mask[None, :], other=0.0)
        qk = tl.dot(query, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            # mask = (offs_m[:, None] >= offs_n[None, :])
            mask = (offs_m[:, None] >= offs_n[None, :]) & (offs_m < N_CTX)[:, None] & (offs_n < N_CTX)[None, :]

            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              kv_stride_z, kv_stride_h,  #
              H, N_CTX,  #
              kv_head_num, # 
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    batch_id = (bhid // H)
    adj = (stride_h * (bhid % H) + stride_z * batch_id).to(tl.int64)
    kv_adj = (kv_stride_h * (bhid % kv_head_num) + kv_stride_z * batch_id).to(tl.int64)

    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += kv_adj
    V += kv_adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, BLOCK_DMODEL)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_n_mask = offs_n < N_CTX

    dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    key = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_n_mask[:, None], other=0.0)
    value = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_n_mask[:, None], other=0.0)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv = _attn_bwd_dkdv(dk, dv,  #
                            Q, key, value, sm_scale,  #
                            DO,  #
                            M, D,  #
                            stride_tok, stride_d,  #
                            H, N_CTX,  #
                            MASK_BLOCK_M1, BLOCK_N1, BLOCK_DMODEL,  #
                            start_n, start_m, num_steps,  #
                            MASK=True  #
                            )

    start_m += num_steps * MASK_BLOCK_M1
    # num_steps = (N_CTX - start_m) // BLOCK_M1
    num_steps = tl.cdiv((N_CTX - start_m), BLOCK_M1)


    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        Q, key, value, sm_scale,  #
        DO,  #
        M, D,  #
        stride_tok, stride_d,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, BLOCK_DMODEL,  #
        start_n, start_m, num_steps,  #
        MASK=False  #
    )
    # tl.device_print("dv: ", dv)


    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv, mask=offs_n_mask[:, None])

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk, mask=offs_n_mask[:, None])

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    query = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, BLOCK_DMODEL], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(dq, query, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, MASK_BLOCK_N2, BLOCK_DMODEL,  #
                      start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                      MASK=True  #
                      )
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(dq, query, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, BLOCK_DMODEL,  #
                      start_m, end_n - num_steps * BLOCK_N2, num_steps,  #
                      MASK=False  #
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    # tl.store(dq_ptrs, dq)

    offs_m_mask = offs_m < N_CTX
    tl.store(dq_ptrs, dq, mask=offs_m_mask[:, None])


class ScaleDotProductAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, query, key, value, attn_mask, causal, sm_scale):
        logging.debug("GEMS SCALED DOT PRODUCT ATTENTION")
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
        # when value is in float8_e5m2 it is transposed.
        HEAD_DIM_V = value.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        # assert dropout_p == 0.0, "Currenty only support dropout_p=0.0"

        o = torch.empty_like(query, dtype=value.dtype)

        stage = 3 if is_causal else 1

        kv_head_num = key.shape[1]

        if sm_scale is None:
            sm_scale = 1.0 / (HEAD_DIM_K**0.5)

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


        # grid = (triton.cdiv(query.shape[2], BLOCK_M), query.shape[0] * query.shape[1], 1)

        grid = lambda args: (
            triton.cdiv(query.shape[2], args["BLOCK_M"]),
            query.shape[0] * query.shape[1],
            1,
        )

        M = torch.empty((query.shape[0], query.shape[1], query.shape[2]), device=query.device, dtype=torch.float32)

        # with torch_device_fn.device(query.device):
        _attn_fwd[grid](
            query,
            key,
            value,
            attn_mask,
            sm_scale,
            M, 
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

        ctx.save_for_backward(query, key, value, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = HEAD_DIM_K
        ctx.causal = causal
        ctx.kv_head_num = kv_head_num
        return o

    @staticmethod
    def backward(ctx, do):
        query, key, value, o, M = ctx.saved_tensors
        batch, num_head, seq_len, head_size = query.shape 
        group_head = num_head // ctx.kv_head_num

        assert do.is_contiguous()
        # assert query.stride() == key.stride() == value.stride() == o.stride() == do.stride()

        assert key.stride() == value.stride()

        # logger.info(f"{do.shape=}")
        # tmp_do = do 
        # do = torch.zeros((1, 1, 256, 128), device="cuda", dtype=torch.float16)
        # do[:, :, :192, :] = tmp_do 

        # dq = torch.empty_like(query)
        # dk = torch.empty_like(key)
        # dv = torch.empty_like(value)

        dq = torch.empty_like(query)
        dk = torch.empty_like(query)
        dv = torch.empty_like(query)


        BATCH, N_HEAD, N_CTX = query.shape[:3]

        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        # RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)

        RCP_LN2 = 1.0 / math.log(2)


        arg_k = key
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        # PRE_BLOCK = 128
        PRE_BLOCK = 256

        # PRE_BLOCK = 32
        # assert N_CTX % PRE_BLOCK == 0
        # pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)

        pre_grid = (triton.cdiv(N_CTX, PRE_BLOCK), BATCH * N_HEAD)


        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, D_HEAD=ctx.BLOCK_DMODEL  #
        )
        # grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)

        grid = (triton.cdiv(N_CTX, BLOCK_N1), 1, BATCH * N_HEAD)
        logger.info(f"{triton.cdiv(N_CTX, BLOCK_N1)=}")
        logger.info(f"{M.shape=}")

        _attn_bwd[grid](
            query, arg_k, value, ctx.sm_scale, do, dq, dk, dv,  #
            M, delta,  #
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),  #
            key.stride(0), key.stride(1),  #
            N_HEAD, N_CTX,  #
            ctx.kv_head_num, # 
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )

        print("group_head is: ", group_head)
        if group_head > 1: 
            dk = dk.reshape(batch, num_head // group_head, group_head, seq_len, head_size)
            dv = dv.reshape(batch, num_head // group_head, group_head, seq_len, head_size)
            dk = dk.sum(dim=2)
            dv = dv.sum(dim=2)

        return dq, dk, dv, None, None, None


from copy import deepcopy 

if __name__ == "__main__": 
    batch = 1
    q_num_head = 8
    kv_num_head = 1
    # seq_len = 192
    # seq_len = 278

    seq_len = 128


    head_size = 128

    torch.manual_seed(0)
    import numpy as np 
    np.random.seed(0)
    np_query = np.random.uniform(-0.2, 0.2, (batch, q_num_head, seq_len, head_size))
    np_key = np.random.uniform(-0.2, 0.2, (batch, kv_num_head, seq_len, head_size))
    np_value = np.random.uniform(-0.2, 0.2, (batch, kv_num_head, seq_len, head_size))

    # query = torch.randn((batch, num_head, seq_len, head_size), device="cuda", dtype=torch.float16, requires_grad=True)
    # key = torch.randn((batch, num_head, seq_len, head_size), device="cuda", dtype=torch.float16, requires_grad=True)
    # value = torch.randn((batch, num_head, seq_len, head_size), device="cuda", dtype=torch.float16, requires_grad=True)

    query = torch.tensor(np_query, device="cuda", dtype=torch.float16, requires_grad=True)
    key = torch.tensor(np_key, device="cuda", dtype=torch.float16, requires_grad=True)
    value = torch.tensor(np_value, device="cuda", dtype=torch.float16, requires_grad=True)
    
    # ref_query = query.clone()
    # ref_query.requires_grad = True
    # ref_key = key.clone()
    # ref_key.requires_grad = True

    # ref_value = value.clone()
    # ref_value.requires_grad = True


    is_causal = True 
    sm_scale = 1.3
    triton_result = ScaleDotProductAttention.apply(query, key, value, None, is_causal, sm_scale)


    torch_result = torch.nn.functional.scaled_dot_product_attention(
        # ref_query,
        # ref_key,
        # ref_value,
        query, 
        key, 
        value, 
        scale=sm_scale,
        is_causal=is_causal,
    )

    print("triton result is: ", triton_result)

    print("torch result is: ", torch_result)


    dout = torch.randn_like(query)

    torch_result.backward(dout)
    torch_q_grad, query.grad = query.grad.clone(), None 
    torch_k_grad, key.grad = key.grad.clone(), None 
    torch_v_grad, value.grad = value.grad.clone(), None 

    triton_result.backward(dout)
    triton_q_grad, query.grad = query.grad.clone(), None 
    triton_k_grad, key.grad = key.grad.clone(), None 
    triton_v_grad, value.grad = value.grad.clone(), None 


    print("torch query grad is: ", torch_q_grad)
    print("triton query grad is: ", triton_q_grad)
    torch.testing.assert_close(torch_q_grad, triton_q_grad, atol=2e-3, rtol=2e-3)

    print("torch key grad is: ", torch_k_grad)
    print("triton key grad is: ", triton_k_grad)
    # torch.testing.assert_close(torch_k_grad, triton_k_grad, atol=2e-3, rtol=2e-3)

    torch.testing.assert_close(torch_k_grad, triton_k_grad, atol=2e-3, rtol=2e-3)


    # print("torch value grad is: ", torch_v_grad)
    # print("triton value grad is: ", triton_v_grad)
    torch.testing.assert_close(torch_v_grad, triton_v_grad, atol=2e-3, rtol=2e-3)

    

    # print("torch key grad is: ", torch_k_grad)
    # print("triton key grad is: ", triton_k_grad)