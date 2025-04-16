import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

from .accuracy_utils import gems_assert_close, to_reference
from .conftest import TO_CPU

device = flag_gems.device


def make_input(batch, num_head, q_seq_len, kv_seq_len, head_size, dtype):
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

    query = torch.tensor(np_query, device="cuda", dtype=dtype)
    key = torch.tensor(np_key, device="cuda", dtype=dtype)
    value = torch.tensor(np_value, device="cuda", dtype=dtype)

    return query, key, value


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
    query, key, value = make_input(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype
    )
    ref_query = to_reference(query, False)
    ref_key = to_reference(key, False)
    ref_value = to_reference(value, False)

    scale = float(1.0 / np.sqrt(head_size))
    torch_result = torch.nn.functional.scaled_dot_product_attention(
        ref_query,
        ref_key,
        ref_value,
        attn_mask=None,
        scale=scale,
        is_causal=is_causal,
    )

    flaggem_result = flag_gems.ops.scaled_dot_product_attention(
        query, key, value, attn_mask=None, scale=scale, is_causal=is_causal
    )

    gems_assert_close(flaggem_result, torch_result, dtype)


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
    query, key, value = make_input(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype
    )
    ref_query = to_reference(query, False)
    ref_key = to_reference(key, False)
    ref_value = to_reference(value, False)

    scale = float(1.0 / np.sqrt(head_size))
    torch_result = torch.nn.functional.scaled_dot_product_attention(
        ref_query,
        ref_key,
        ref_value,
        attn_mask=None,
        scale=scale,
        is_causal=is_causal,
    )

    with flag_gems.use_gems():
        flaggem_result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, scale=scale, is_causal=is_causal
        )

    gems_assert_close(flaggem_result, torch_result, dtype)


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
    query, key, value = make_input(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype
    )
    ref_query = to_reference(query, False)
    ref_key = to_reference(key, False)
    ref_value = to_reference(value, False)

    scale = float(1.0 / np.sqrt(head_size))
    torch_result = torch.nn.functional.scaled_dot_product_attention(
        ref_query,
        ref_key,
        ref_value,
        attn_mask=None,
        scale=scale,
        is_causal=is_causal,
    )

    with flag_gems.use_gems():
        flaggem_result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, scale=scale, is_causal=is_causal
        )

    gems_assert_close(flaggem_result, torch_result, dtype)


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
    query, key, value = make_input(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype
    )
    ref_query = to_reference(query, False)
    ref_key = to_reference(key, False)
    ref_value = to_reference(value, False)

    scale = float(1.0 / np.sqrt(head_size))

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
        )
        flaggem_result = out.transpose(1, 2)

    gems_assert_close(flaggem_result, torch_result, dtype)


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
    query, key, value = make_input(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype
    )
    scale = float(1.0 / np.sqrt(head_size))

    with flag_gems.use_gems():
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        return_debug_mask = True
        dropout_p = 0.2
        device = torch_device_fn.current_device()
        gen = torch_device_fn.default_generators[device]
        old_state = gen.get_state()
        new_state = torch.tensor((12345, 0), dtype=torch.int64)
        gen.set_state(new_state.view(torch.uint8))
        (
            out,
            lse,
            seed_0,
            offset_0,
            debug_softmax_0,
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
        )
        gen.set_state(new_state.view(torch.uint8))
        (
            out,
            lse,
            seed_1,
            offset_1,
            debug_softmax_1,
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
        )
        gen.set_state(old_state)
        gems_assert_close(debug_softmax_0, debug_softmax_1, dtype)
        dropout_ratio = torch.sum(debug_softmax_0 < 0) / torch.sum(debug_softmax_0 != 0)
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
    query, key, value = make_input(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype
    )

    ref_query = to_reference(query, False)
    ref_key = to_reference(key, False)
    ref_value = to_reference(value, False)

    scale = float(1.0 / np.sqrt(head_size))
    dropout_p = 0
    return_debug_mask = False

    q = ref_query.transpose(1, 2)
    k = ref_key.transpose(1, 2)
    v = ref_value.transpose(1, 2)
    out, lse, _, _, _ = torch.ops.aten._flash_attention_forward(
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
    )
    torch_result = out.transpose(1, 2)
    torch_lse = lse.transpose(1, 2)

    with flag_gems.use_gems():
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out, lse, _, _, _ = torch.ops.aten._flash_attention_forward(
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
        )
        flaggem_result = out.transpose(1, 2)
        flaggem_lse = lse.transpose(1, 2)

    gems_assert_close(flaggem_result, torch_result, dtype)
    gems_assert_close(flaggem_lse, torch_lse, torch.float)
