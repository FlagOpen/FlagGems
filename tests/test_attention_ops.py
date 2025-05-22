import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import set_philox_state

from .accuracy_utils import gems_assert_close, to_reference
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
        **extra_kwargs
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
            **extra_kwargs
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
