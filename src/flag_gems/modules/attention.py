import logging
from typing import Optional

import torch

import flag_gems

logger = logging.getLogger(__name__)

try:
    from flag_gems import ext_ops  # noqa: F401

    has_c_extension = True
except ImportError:
    has_c_extension = False


def gems_flash_attention_impl_forwad(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,  #: FlashAttentionMetadata,
    output: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward pass with FlashAttention.

    Args:
        query: shape = [num_tokens, num_heads, head_size]
        key: shape = [num_tokens, num_kv_heads, head_size]
        value: shape = [num_tokens, num_kv_heads, head_size]
        kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: Metadata for attention.
    Returns:
        shape = [num_tokens, num_heads * head_size]
    NOTE: FP8 quantization, flash-attn expect the size of
            {q,k,v}_descale to be (num_sequences, num_kv_heads).
            We use torch's .expand() to avoid duplicating values
    """
    assert output is not None, "Output tensor must be provided."

    if output_scale is not None:
        raise NotImplementedError(
            "fused output quantization is not yet supported" " for FlashAttentionImpl"
        )

    if attn_metadata is None:
        # Profiling run.
        return output

    # IMPORTANT!
    # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
    # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
    # in this method. For example, `view` and `slice` (or `[:n]`) operations
    # are surprisingly slow even in the case they do not invoke any GPU ops.
    # Minimize the PyTorch ops in this method as much as possible.
    # Whenever making a change in this method, please benchmark the
    # performance to make sure it does not introduce any overhead.

    num_actual_tokens = attn_metadata.num_actual_tokens
    key_cache, value_cache = kv_cache.unbind(0)

    if self.kv_sharing_target_layer_name is None:
        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        flag_gems.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    # TODO: Support FP8 quantization.
    if self.kv_cache_dtype.startswith("fp8"):
        raise NotImplementedError(
            "FP8 quantization is not yet supported for FlashAttentionImpl"
        )
        # key_cache = key_cache.view(torch.float8_e4m3fn)
        # value_cache = value_cache.view(torch.float8_e4m3fn)
        # num_tokens, num_heads, head_size = query.shape
        # query, _ = ops.scaled_fp8_quant(
        #     query.reshape((num_tokens, num_heads * head_size)).contiguous(),
        #     layer._q_scale,
        # )
        # query = query.reshape((num_tokens, num_heads, head_size))

    # Compute attention and update output up to `num_actual_tokens`.
    use_local_attn = self.use_irope and attn_metadata.local_attn_metadata is not None

    if not attn_metadata.use_cascade or use_local_attn:
        if use_local_attn:
            assert attn_metadata.local_attn_metadata is not None
            local_metadata = attn_metadata.local_attn_metadata
            cu_seqlens_q = local_metadata.local_query_start_loc
            seqused_k = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
            max_seqlen_k = local_metadata.local_max_seq_len
            block_table = local_metadata.local_block_table
            # scheduler_metadata = local_metadata.local_scheduler_metadata
        else:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            # scheduler_metadata = attn_metadata.scheduler_metadata

        # descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

        flag_gems.flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            # scheduler_metadata=scheduler_metadata,
            # fa_version=self.vllm_flash_attn_version,
            # q_descale=layer._q_scale.expand(descale_shape),
            # k_descale=layer._k_scale.expand(descale_shape),
            # v_descale=layer._v_scale.expand(descale_shape),
        )
        return output

    # TODO: Support cascade attention.
    raise NotImplementedError(
        "Cascade attention is not implemented in this version of vLLM."
    )

    # assert not use_local_attn, "Cascade attention does not support local attention."
    # # Cascade attention (rare case).
    # cascade_attention(
    #     output[:num_actual_tokens],
    #     query[:num_actual_tokens],
    #     key_cache,
    #     value_cache,
    #     cu_query_lens=attn_metadata.query_start_loc,
    #     max_query_len=attn_metadata.max_query_len,
    #     cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
    #     prefix_kv_lens=attn_metadata.prefix_kv_lens,
    #     suffix_kv_lens=attn_metadata.suffix_kv_lens,
    #     max_kv_len=attn_metadata.max_seq_len,
    #     softmax_scale=self.scale,
    #     alibi_slopes=self.alibi_slopes,
    #     sliding_window=self.sliding_window,
    #     logits_soft_cap=self.logits_soft_cap,
    #     block_table=attn_metadata.block_table,
    #     common_prefix_len=attn_metadata.common_prefix_len,
    #     fa_version=self.vllm_flash_attn_version,
    #     prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
    #     suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
    #     q_descale=layer._q_scale,
    #     k_descale=layer._k_scale,
    #     v_descale=layer._v_scale,
    # )
    # return output
