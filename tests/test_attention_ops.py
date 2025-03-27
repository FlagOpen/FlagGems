import numpy as np
import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference


@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize(["batch", "num_head", "q_seq_len", "kv_seq_len"],
                         [
                          (1, 1, 128, 2048),
                          (4, 8, 1024, 1024),
                          (4, 8, 1024, 128),
                          (4, 8, 17, 1030)
                          ])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("add_bias", [False])
@pytest.mark.parametrize("is_causal", [False, True])
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

    query = torch.tensor(np_query, device="cuda", dtype=dtype)
    key = torch.tensor(np_key, device="cuda", dtype=dtype)
    value = torch.tensor(np_value, device="cuda", dtype=dtype)
    if add_bias:
        attn_bias = torch.tensor(np_attn_bias, device="cuda", dtype=dtype)
    else:
        attn_bias = None

    ref_query = to_reference(query, False)
    ref_key = to_reference(key, False)
    ref_value = to_reference(value, False)
    ref_attn_bias = to_reference(attn_bias, False) if add_bias else None

    scale = float(1.0 / np.sqrt(head_size))

    if is_causal and q_seq_len != kv_seq_len:
        # Pytorch treats non-square causal as a special case where the sdp attention
        # does not route to flash attn and instead uses mem-eff attn.
        # In this case, we directly compare on the lower level _flash_attention_forward
        q = ref_query.transpose(1, 2)
        k = ref_key.transpose(1, 2)
        v = ref_value.transpose(1, 2)
        out, *_ = torch.ops.aten._flash_attention_forward(
            q,
            k,
            v,
            None,
            None,
            q.shape[-3],
            k.shape[-3],
            0,
            is_causal,
            False,
            scale=scale,
            window_size_left=None,
            window_size_right=None,
            seqused_k=None,
            alibi_slopes=None
        )
        torch_result = out.transpose(1, 2)
        
        with flag_gems.use_gems():
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            out, *_ = torch.ops.aten._flash_attention_forward(
                q,
                k,
                v,
                None,
                None,
                q.shape[-3],
                k.shape[-3],
                0,
                is_causal,
                False,
                scale=scale,
                window_size_left=None,
                window_size_right=None,
                seqused_k=None,
                alibi_slopes=None
            )
            flaggem_result = out.transpose(1, 2)
    else:
        attn_mask = None if is_causal else ref_attn_bias

        torch_result = torch.nn.functional.scaled_dot_product_attention(
            ref_query,
            ref_key,
            ref_value,
            attn_mask=ref_attn_bias,
            scale=scale,
            is_causal=is_causal,
        )

        with flag_gems.use_gems():
            attn_mask = None if is_causal else attn_bias
            flaggem_result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_bias, scale=scale, is_causal=is_causal
            )

    gems_assert_close(flaggem_result, torch_result, dtype)


@pytest.mark.flash_attention_forward
@pytest.mark.parametrize(["batch", "num_head", "q_seq_len", "kv_seq_len"],
                         [
                          (1, 1, 128, 2048),
                          (8, 32, 1024, 1024),
                          (8, 32, 1024, 128),
                          (8, 32, 17, 1030)
                          ])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(["window_size_left", "window_size_right"],
                         [
                            (None, None),
                            (256, 0),
                            (128, 128)
                          ])
@pytest.mark.parametrize("is_causal", [False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attention_forward(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, window_size_left, window_size_right, dtype
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

    query = torch.tensor(np_query, device="cuda", dtype=dtype)
    key = torch.tensor(np_key, device="cuda", dtype=dtype)
    value = torch.tensor(np_value, device="cuda", dtype=dtype)

    ref_query = to_reference(query, False)
    ref_key = to_reference(key, False)
    ref_value = to_reference(value, False)

    scale = float(1.0 / np.sqrt(head_size))
    dropout_p = 0
    return_debug_mask = False

    q = ref_query.transpose(1, 2)
    k = ref_key.transpose(1, 2)
    v = ref_value.transpose(1, 2)
    out, lse, _, _, _  = torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        seqused_k=None,
        alibi_slopes=None
    )
    torch_result = out.transpose(1, 2)
    torch_lse = lse.transpose(1, 2)
    
    with flag_gems.use_gems():
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out, lse, _, _, _  = torch.ops.aten._flash_attention_forward(
            q,
            k,
            v,
            None,
            None,
            q.shape[-3],
            k.shape[-3],
            dropout_p,
            is_causal,
            return_debug_mask,
            scale=scale,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            seqused_k=None,
            alibi_slopes=None
        )
        flaggem_result = out.transpose(1, 2)
        flaggem_lse = lse.transpose(1, 2)

    gems_assert_close(flaggem_result, torch_result, dtype)
    gems_assert_close(flaggem_lse, torch_lse, torch.float)