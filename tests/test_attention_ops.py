import numpy as np
import pytest
import torch
import triton
import math

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

device = flag_gems.device


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.device == "musa", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize("batch", [8, 16])
@pytest.mark.parametrize("num_head", [1, 8])
@pytest.mark.parametrize("q_seq_len", [17, 64, 128])
@pytest.mark.parametrize("kv_seq_len", [7, 87, 128, 577, 2048])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("add_bias", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_dot_product_attention(
    batch, num_head, q_seq_len, kv_seq_len, head_size, add_bias, is_causal, dtype
):
    np.random.seed(0)
    np_query = np.random.uniform(
        -0.05, 0.05, (batch, num_head, q_seq_len, head_size)
    ).astype(np.float32)
    np_key = np.random.uniform(
        -0.05, 0.05, (batch, num_head, kv_seq_len, head_size)
    ).astype(np.float32)
    np_value = np.random.uniform(
        -0.05, 0.05, (batch, num_head, kv_seq_len, head_size)
    ).astype(np.float32)
    np_attn_bias = np.random.uniform(
        -0.05, 0.05, (batch, num_head, q_seq_len, kv_seq_len)
    ).astype(np.float32)

    query = torch.tensor(np_query, device=device, dtype=dtype)
    key = torch.tensor(np_key, device=device, dtype=dtype)
    value = torch.tensor(np_value, device=device, dtype=dtype)
    if add_bias:
        attn_bias = torch.tensor(np_attn_bias, device=device, dtype=dtype)
    else:
        attn_bias = None

    ref_query = to_reference(query, False)
    ref_key = to_reference(key, False)
    ref_value = to_reference(value, False)
    ref_attn_bias = to_reference(attn_bias, False) if add_bias else None

    scale = float(1.0 / np.sqrt(head_size))

    if is_causal:
        torch_result = torch.nn.functional.scaled_dot_product_attention(
            ref_query,
            ref_key,
            ref_value,
            scale=scale,
            is_causal=is_causal,
        )
    else:
        torch_result = torch.nn.functional.scaled_dot_product_attention(
            ref_query,
            ref_key,
            ref_value,
            attn_mask=ref_attn_bias,
            scale=scale,
            is_causal=is_causal,
        )

    with flag_gems.use_gems():
        if is_causal:
            flaggem_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, scale=scale, is_causal=is_causal
            )
        else:
            flaggem_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_bias, scale=scale, is_causal=is_causal
            )
    gems_assert_close(flaggem_result, torch_result, dtype, reduce_dim=head_size)



# @pytest.mark.skipif(TO_CPU, reason="CPU not supported now")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.device == "musa", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.flash_mla
@pytest.mark.parametrize("seqlen", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16,])
def test_flash_mla(
    seqlen, dtype
):
    b = 128
    s_q = 1
    h_q = 128
    h_kv = 1
    d = 576
    dv = 512
    causal = True
    block_size = 64
    cache_seqlens = torch.tensor([seqlen + 2 * i for i in range(b)], dtype=torch.int32, device=device)
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256

    q = torch.randn([b, s_q, h_q, d], dtype=dtype, device=device)
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32, device=device).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn([block_table.numel(), block_size, h_kv, d], dtype=dtype, device=device)
    blocked_v = blocked_k[..., :dv]

    ref_q = to_reference(q)
    ref_block_table = to_reference(block_table)
    ref_blocked_k = to_reference(blocked_k)
    ref_blocked_v = to_reference(blocked_v)
    ref_cache_seqlens = to_reference(cache_seqlens)


    def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
        query = query.float()
        key = key.float()
        value = value.float()
        key = key.repeat_interleave(h_q // h_kv, dim=0)
        value = value.repeat_interleave(h_q // h_kv, dim=0)
        attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        if is_causal:
            s_q = query.shape[-2]
            s_k = key.shape[-2]
            attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device=query.device)
            temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device=query.device).tril(diagonal=s_k - s_q)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)
            attn_weight += attn_bias
        lse = attn_weight.logsumexp(dim=-1)
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        return attn_weight @ value, lse

    def ref_mla(
        q,
        block_table,
        blocked_k,
        blocked_v,
        max_seqlen_pad,
        block_size,
        b,
        s_q,
        cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        causal,
        dtype,
    ):
        device = q.device
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device=device)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=device)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse
    
    ref_out, _ = ref_mla(
        ref_q,
        ref_block_table,
        ref_blocked_k,
        ref_blocked_v,
        max_seqlen_pad,
        block_size,
        b,
        s_q,
        ref_cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        causal,
        dtype,
    )
    res_out = flag_gems.flash_mla(
        q,
        block_table,
        blocked_k,
        max_seqlen_pad,
        block_size,
        b,
        s_q,
        cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        causal,
        dtype,
    )


    def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
        x, y = x.double(), y.double()
        x = x.to(y.device)
        RMSE = ((x - y) * (x - y)).mean().sqrt().item()
        cos_diff = 1 - 2 * (x * y).sum().item() / max(
            (x * x + y * y).sum().item(), 1e-12
        )
        amax_diff = (x - y).abs().max().item()
        assert cos_diff < 1e-5, f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}"

    cal_diff(res_out, ref_out, "out")