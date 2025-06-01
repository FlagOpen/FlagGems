import math

import torch
import triton

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import update_philox_state

from .flash_kernel import (
    block_m_splitkv_heuristic,
    block_n_splitkv_heuristic,
    flash_fwd_kernel,
    flash_fwd_splitkv_combine_kernel,
    flash_fwd_splitkv_kernel,
    flash_varlen_fwd_kernel,
)

_debug = False


def CHECK_DEVICE(x):
    assert x.device.type == flag_gems.device


class params_varlen_fwd:
    __slots__ = (
        # pointers and strides
        "q_ptr",
        "k_ptr",
        "v_ptr",
        "o_ptr",
        "p_ptr",
        "softmax_lse_ptr",
        "q_row_stride",
        "k_row_stride",
        "v_row_stride",
        "q_head_stride",
        "k_head_stride",
        "v_head_stride",
        "o_row_stride",
        "o_head_stride",
        "q_batch_stride",
        "k_batch_stride",
        "v_batch_stride",
        "o_batch_stride",
        "is_cu_seqlens_q",
        "cu_seqlens_q_ptr",
        "is_cu_seqlens_k",
        "cu_seqlens_k_ptr",
        "is_seqused_k",
        "seqused_k_ptr",
        # sizes
        "b",
        "bk",
        "h",
        "hk",
        "h_hk_ratio",
        "seqlen_q",
        "seqlen_k",
        "seqlen_q_rounded",
        "seqlen_v_rounded",
        "d",
        "d_rounded",
        # scaling factors
        "softcap",
        "scale_softmax",
        "scale_softmax_log2",
        "is_causal",
        "is_local",
        "window_size_left",
        "window_size_right",
        "seqlenq_ngroups_swapped",
        # alibi
        "is_alibi",
        "alibi_slopes_ptr",
        "alibi_slopes_batch_stride",
        # block table
        "total_q",
        "page_table_ptr",
        "page_table_batch_stride",
        "block_size",
    )

    def __init__(
        self,
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        p_ptr,
        softmax_lse_ptr,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        q_head_stride,
        k_head_stride,
        v_head_stride,
        o_row_stride,
        o_head_stride,
        q_batch_stride,
        k_batch_stride,
        v_batch_stride,
        o_batch_stride,
        is_cu_seqlens_q,
        cu_seqlens_q_ptr,
        is_cu_seqlens_k,
        cu_seqlens_k_ptr,
        is_seqused_k,
        seqused_k_ptr,
        # sizes
        b,
        bk,
        h,
        hk,
        h_hk_ratio,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        seqlen_v_rounded,
        d,
        d_rounded,
        # scaling factors
        softcap,
        scale_softmax,
        scale_softmax_log2,
        is_causal,
        is_local,
        window_size_left,
        window_size_right,
        seqlenq_ngroups_swapped,
        # alibi
        is_alibi,
        alibi_slopes_ptr,
        alibi_slopes_batch_stride,
        # block table
        total_q,
        page_table_ptr,
        page_table_batch_stride,
        block_size,
    ):
        self.q_ptr = q_ptr
        self.k_ptr = k_ptr
        self.v_ptr = v_ptr
        self.o_ptr = o_ptr
        self.p_ptr = p_ptr
        self.softmax_lse_ptr = softmax_lse_ptr
        self.q_row_stride = q_row_stride
        self.k_row_stride = k_row_stride
        self.v_row_stride = v_row_stride
        self.q_head_stride = q_head_stride
        self.k_head_stride = k_head_stride
        self.v_head_stride = v_head_stride
        self.o_row_stride = o_row_stride
        self.o_head_stride = o_head_stride
        self.q_batch_stride = q_batch_stride
        self.k_batch_stride = k_batch_stride
        self.v_batch_stride = v_batch_stride
        self.o_batch_stride = o_batch_stride
        self.is_cu_seqlens_q = is_cu_seqlens_q
        self.cu_seqlens_q_ptr = cu_seqlens_q_ptr
        self.is_cu_seqlens_k = is_cu_seqlens_k
        self.cu_seqlens_k_ptr = cu_seqlens_k_ptr
        self.is_seqused_k = is_seqused_k
        self.seqused_k_ptr = seqused_k_ptr
        # sizes
        self.b = b
        self.bk = bk
        self.h = h
        self.hk = hk
        self.h_hk_ratio = h_hk_ratio
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.seqlen_q_rounded = seqlen_q_rounded
        self.seqlen_v_rounded = seqlen_v_rounded
        self.d = d
        self.d_rounded = d_rounded
        # scaling factors
        self.softcap = softcap
        self.scale_softmax = scale_softmax
        self.scale_softmax_log2 = scale_softmax_log2
        self.is_causal = is_causal
        self.is_local = is_local
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped
        # alibi
        self.is_alibi = is_alibi
        self.alibi_slopes_ptr = alibi_slopes_ptr
        self.alibi_slopes_batch_stride = alibi_slopes_batch_stride
        # block table
        self.total_q = total_q
        self.page_table_ptr = page_table_ptr
        self.page_table_batch_stride = page_table_batch_stride
        self.block_size = block_size


def mha_varlan_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    page_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
):
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    q_device = q.device
    q_dtype = q.dtype
    assert q_dtype in (
        torch.float16,
        torch.bfloat16,
    ), "FlashAttention only support fp16 and bf16 data type"
    assert q_dtype == k.dtype
    assert q_dtype == v.dtype
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"

    assert cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_q.is_contiguous()

    assert cu_seqlens_k.dtype == torch.int32
    assert cu_seqlens_k.is_contiguous()

    assert page_table is not None

    # q shape: [total_q_tokens, num_heads, head_size]
    # k shape:
    #   paged_kv: [num_pages, block_size, num_heads_k, head_size]
    # batch_size, number of sentences
    total_q, num_heads, head_size = q.size()
    num_heads_k = k.size(2)
    batch_size = cu_seqlens_q.numel() - 1
    block_size = k.size(1)
    num_pages = k.size(0)
    k_batch_size = num_pages
    # max_num_pages_per_seq = page_table.size(1)
    page_table_batch_stride = page_table.stride(0)
    k_batch_stride = k.stride(0)
    v_batch_stride = v.stride(0)

    assert k.size() == v.size()
    assert cu_seqlens_q.size() == (batch_size + 1,)
    assert cu_seqlens_k.size() == (batch_size + 1,)

    if seqused_k is not None:
        assert seqused_k.is_contiguous()
        assert seqused_k.size() == (batch_size,)

    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False

    if is_causal:
        window_size_right = 0

    # check disable swa
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_k:
        window_size_right = -1

    is_local = window_size_left >= 0

    # Optimize all single-query sequences by swapping the query-group and sequence dimensions
    seqlenq_ngroups_swapped = (
        max_seqlen_q == 1
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
        and p_dropout == 0
    )
    q_groups = num_heads / num_heads_k
    if seqlenq_ngroups_swapped:
        q = (
            q.reshape((batch_size, num_heads_k, q_groups, head_size))
            .transpose(1, 2)
            .reshape(batch_size * q_groups, num_heads_k, head_size)
        )
        max_seqlen_q = q_groups
        num_heads = num_heads_k
        cu_seqlens_q = None
        q_batch_stride = q.stride(0) * max_seqlen_q
        k_batch_stride = k.stride(0)
        v_batch_stride = v.stride(0)
        o_batch_stride = out.stride(0) * max_seqlen_q
    else:
        q_batch_stride = 0
        k_batch_stride = 0
        v_batch_stride = 0
        o_batch_stride = 0

    total_q = q.size(0)

    assert leftpad_k is None, "leftpad_k is not supported."
    assert (
        head_size <= 256
    ), "FlashAttention forward only supports head dimension at most 256"
    assert (
        head_size % 8 == 0
    ), "head_size must be a multiple of 8, this is ensured by padding!"
    assert (
        num_heads % num_heads_k == 0
    ), "Number of heads in key/value must divide number of heads in query"

    assert q.shape == (total_q, num_heads, head_size)
    assert k.shape == (num_pages, block_size, num_heads_k, head_size)
    assert v.shape == (num_pages, block_size, num_heads_k, head_size)
    assert k.stride() == v.stride()

    if softcap > 0:
        assert p_dropout == 0, "dropout is not supported if softcap is used."

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32) if head_size < 192 else 256
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 32)

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
        is_alibi = True
    else:
        alibi_slopes_batch_stride = 0
        is_alibi = False

    # Prepare params to kernel
    with torch_device_fn.device(q_device):
        if out is not None:
            assert out.stride(-1) == 1
            assert out.dtype == q.dtype
            assert out.size() == (total_q, num_heads, head_size)
            out_ = out
            if seqlenq_ngroups_swapped:
                out = torch.empty_like(q, dtype=v.dtype)
        else:
            out_ = None
            out = torch.empty_like(q, dtype=v.dtype)
        lse = torch.empty((num_heads, total_q), dtype=torch.float, device=q_device)

        assert return_softmax is False, "not supported."

        p = torch.empty((), device=q_device)

        assert p_dropout == 0, "dropout is not supported."

        if zero_tensors:
            out.zero_()
            lse.fill_(float("-inf"))

        params = params_varlen_fwd(
            q,  # q_ptr,
            k,  # k_ptr,
            v,  # v_ptr,
            out,  # o_ptr,
            p,  # p_ptr,
            lse,  # softmax_lse_ptr,
            q.stride(-3),  # q_row_stride,
            k.stride(-3),  # k_row_stride,
            v.stride(-3),  # v_row_stride,
            q.stride(-2),  # q_head_stride,
            k.stride(-2),  # k_head_stride,
            v.stride(-2),  # v_head_stride,
            out.stride(-3),  # o_row_stride,
            out.stride(-2),  # o_head_stride,
            q_batch_stride,  # q_batch_stride,
            k_batch_stride,  # k_batch_stride,
            v_batch_stride,  # v_batch_stride,
            o_batch_stride,  # o_batch_stride,
            cu_seqlens_q is not None,  # is_cu_seqlens_q,
            cu_seqlens_q,  # cu_seqlens_q_ptr,
            seqused_k is None,  # is_cu_seqlens_k,
            cu_seqlens_k,  # cu_seqlens_k_ptr,
            seqused_k is not None,  # is_seqused_k,
            seqused_k,  # seqused_k_ptr,
            # sizes
            batch_size,  # b,
            k_batch_size,  # bk,
            num_heads,  # h,
            num_heads_k,  # hk,
            num_heads // num_heads_k,  # h_hk_ratio,
            max_seqlen_q,  # seqlen_q,
            max_seqlen_k,  # seqlen_k,
            seqlen_q_rounded,  # seqlen_q_rounded,
            seqlen_k_rounded,  # seqlen_v_rounded,
            head_size,  # d,
            head_size_rounded,  # d_rounded,
            # scaling factors
            softcap,  # softcap,
            softmax_scale,  # scale_softmax,
            softmax_scale_log2e,  # scale_softmax_log2,
            # causal and swa
            is_causal,  # is_causal,
            is_local,  # is_local,
            window_size_left,  # window_size_left,
            window_size_right,  # window_size_right,
            seqlenq_ngroups_swapped,  # seqlenq_ngroups_swapped,
            # alibi
            is_alibi,  #
            alibi_slopes,  # alibi_slopes_ptr,
            alibi_slopes_batch_stride,  # alibi_slopes_batch_stride,
            # block table params
            total_q,  # total_q,
            page_table,  # page_table_ptr,
            page_table_batch_stride,  # page_table_batch_stride,
            block_size,  # block_size,
        )

        grid = lambda args: (
            triton.cdiv(max_seqlen_q, args["BLOCK_M"]),
            batch_size,
            num_heads,
        )
        kernel = flash_varlen_fwd_kernel[grid]
        args = tuple(getattr(params, k) for k in params.__slots__)

        # We have to forego parameter autotuning and particularly fix BLOCK_N
        # to avoid breaking a kv block onto multiple cache pages.
        BLOCK_M, BLOCK_N = 128, 32
        assert block_size % BLOCK_N == 0, f"block_size must be divisible by {BLOCK_N}."
        kernel(*args, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=3)

        if seqlenq_ngroups_swapped:
            out = out.reshape(
                batch_size, max_seqlen_q, num_heads_k, head_size
            ).transpose(1, 2)
            if out_:
                out_.view(batch_size, num_heads_k, max_seqlen_q, head_size).copy_(out)
                out = out_
            else:
                out = out.reshape(batch_size, num_heads_k * max_seqlen_q, head_size)
            lse = lse.reshape(num_heads_k, batch_size, max_seqlen_q)
            lse = lse.reshape(num_heads_k * max_seqlen_q, batch_size)

    return out, lse


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
            "append_kv": False,
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
