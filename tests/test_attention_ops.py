from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import set_philox_state

from .accuracy_utils import gems_assert_close, init_seed, to_reference
from .conftest import TO_CPU

device = flag_gems.device


def make_input(batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, device):
    set_philox_state(1234567890, 0, device)
    q_shape = (batch, num_head, q_seq_len, head_size)
    kv_shape = (batch, num_head, kv_seq_len, head_size)
    q = torch.empty(q_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    k = torch.empty(kv_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    v = torch.empty(kv_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    return q, k, v


def torch_sdpa(q, k, v, scale, is_causal):
    torch_result = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        scale=scale,
        is_causal=is_causal,
    )
    return torch_result


def torch_flash_fwd(
    q, k, v, scale, is_causal, dropout_p=0, return_debug_mask=False, **extra_kwargs
):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    (
        out,
        lse,
        seed,
        offset,
        debug_softmax,
    ) = torch.ops.aten._flash_attention_forward(
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
        **extra_kwargs,
    )

    return out, lse, seed, offset, debug_softmax


def gems_flash_fwd(
    q, k, v, scale, is_causal, dropout_p=0, return_debug_mask=False, **extra_kwargs
):
    with flag_gems.use_gems():
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        (
            out,
            lse,
            seed,
            offset,
            debug_softmax,
        ) = torch.ops.aten._flash_attention_forward(
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
            **extra_kwargs,
        )

    return out, lse, seed, offset, debug_softmax


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.device == "musa", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [(4, 8, 1024, 1024), (4, 8, 2048, 256), (4, 8, 17, 1030)],
)
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sdpa_legacy(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = to_reference(q, False)
    ref_k = to_reference(k, False)
    ref_v = to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))
    torch_result = torch_sdpa(ref_q, ref_k, ref_v, scale, is_causal)

    gems_result = flag_gems.ops.scaled_dot_product_attention(
        q, k, v, attn_mask=None, scale=scale, is_causal=is_causal
    )

    gems_assert_close(gems_result, torch_result, dtype)


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.device == "musa", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [
        (4, 8, 1024, 1024),
    ],
)
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sdpa_square_qk_even_mn(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = to_reference(q, False)
    ref_k = to_reference(k, False)
    ref_v = to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))
    torch_result = torch_sdpa(ref_q, ref_k, ref_v, scale, is_causal)
    with flag_gems.use_gems():
        gems_result = torch_sdpa(q, k, v, scale, is_causal)
    gems_assert_close(gems_result, torch_result, dtype)


@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.device == "musa", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [(1, 1, 128, 2048), (4, 8, 1024, 128), (4, 8, 17, 1030)],
)
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("is_causal", [False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sdpa_nonsquare_qk(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = to_reference(q, False)
    ref_k = to_reference(k, False)
    ref_v = to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))
    torch_result = torch_sdpa(ref_q, ref_k, ref_v, scale, is_causal)
    with flag_gems.use_gems():
        gems_result = torch_sdpa(q, k, v, scale, is_causal)
    gems_assert_close(gems_result, torch_result, dtype)


@pytest.mark.skipif(TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.device == "musa", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.flash_attention_forward
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [(1, 1, 128, 2048), (4, 8, 1024, 128), (4, 8, 17, 1030)],
)
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("is_causal", [True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_fwd_nonsquare_qk_causal(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = to_reference(q, False)
    ref_k = to_reference(k, False)
    ref_v = to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    torch_out, torch_lse, _, _, _ = torch_flash_fwd(
        ref_q, ref_k, ref_v, scale, is_causal
    )
    gems_out, gems_lse, _, _, _ = gems_flash_fwd(q, k, v, scale, is_causal)

    gems_assert_close(gems_out, torch_out, dtype)
    gems_assert_close(gems_lse, torch_lse, torch.float)


@pytest.mark.skipif(TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.device == "musa", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.flash_attention_forward
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [
        (1, 1, 1024, 1024),
    ],
)
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_fwd_dropout(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    scale = float(1.0 / np.sqrt(head_size))
    dropout_p = 0.2

    gems_out, gems_lse, _, _, debug_softmax = gems_flash_fwd(
        q, k, v, scale, is_causal, dropout_p=dropout_p, return_debug_mask=True
    )

    dropout_ratio = torch.sum(debug_softmax < 0) / torch.sum(debug_softmax != 0)
    np.testing.assert_allclose(dropout_ratio.to("cpu"), dropout_p, rtol=5e-2)


@pytest.mark.skipif(TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(torch.__version__ < "2.4", reason="Low Pytorch Version")
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.device == "musa", reason="RuntimeError")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.flash_attention_forward
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    [(1, 1, 128, 2048), (8, 32, 1024, 1024), (8, 32, 1024, 128), (8, 32, 17, 1030)],
)
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    ["window_size_left", "window_size_right"], [(256, 0), (128, 128)]
)
@pytest.mark.parametrize("is_causal", [False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_fwd_swa(
    batch,
    num_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    window_size_left,
    window_size_right,
    dtype,
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = to_reference(q, False)
    ref_k = to_reference(k, False)
    ref_v = to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    torch_out, torch_lse, _, _, _ = torch_flash_fwd(
        ref_q,
        ref_k,
        ref_v,
        scale,
        is_causal,
        dropout_p=0,
        return_debug_mask=False,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    gems_out, gems_lse, _, _, _ = gems_flash_fwd(
        q,
        k,
        v,
        scale,
        is_causal,
        dropout_p=0,
        return_debug_mask=False,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )

    gems_assert_close(gems_out, torch_out, dtype)
    gems_assert_close(gems_lse, torch_lse, torch.float)


# Following varlen and paged attn tests are copied from
# https://github.com/vllm-project/flash-attention/blob/main/tests/test_vllm_flash_attn.py
def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        # clone to avoid clobbering the query tensor
        q = query[start_idx : start_idx + query_len].clone()
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", [(4, 4), (8, 2), (16, 2)])
@pytest.mark.parametrize("head_size", [128, 256])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", [32768, 2048])
@torch.inference_mode()
def test_varlen_with_paged_kv(
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
) -> None:
    torch.set_default_device(flag_gems.device)
    init_seed(1234567890)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (
        (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)
    )
    scale = head_size**-0.5
    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = flag_gems.ops.flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=seqused_k,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        fa_version=2,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    torch.testing.assert_close(
        output, ref_output, atol=2e-2, rtol=1e-2
    ), f"{torch.max(torch.abs(output - ref_output))}"
