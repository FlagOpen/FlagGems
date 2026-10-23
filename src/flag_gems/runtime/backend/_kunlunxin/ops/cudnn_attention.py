# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kunlunxin (XPU) overrides for the two cuDNN-style attention entry points.

``aten::_cudnn_attention_forward`` and ``aten::_scaled_dot_product_cudnn_attention``
are thin wrappers around ``scaled_dot_product_attention_forward``.  The generic
wrappers import that symbol from ``flag_gems.ops.attention`` -- the *generic*
implementation -- so on Kunlunxin they never reach the tuned XPU FlashAttention
path and return garbage whenever an ``attn_bias`` is supplied (the broadcast
mask test measured differences of 3.4e38 / inf, i.e. the mask was not applied).

These overrides keep the exact signatures and return tuples of the generic
wrappers but call the Kunlunxin forward (``.attention``), which handles
broadcastable masks.
"""

import logging

import torch
from packaging import version

from .attention import scaled_dot_product_attention_forward

logger = logging.getLogger(__name__)


def cudnn_attention_forward(
    query,
    key,
    value,
    attn_bias,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    compute_log_sumexp,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    """Forward kernel for cuDNN attention -- BHSD-native path."""
    logger.debug("GEMS_KUNLUNXIN CUDNN_ATTENTION_FORWARD")

    if dropout_p > 0.0:
        raise NotImplementedError(
            "cudnn_attention_forward: dropout > 0 is not yet supported"
        )
    if cum_seq_q is not None or cum_seq_k is not None:
        raise NotImplementedError(
            "cudnn_attention_forward: varlen is not yet supported"
        )

    # NOTE: the Kunlunxin forward returns a single tensor (the generic one
    # returns ``(out, m_log2)``); unpacking it would slice the batch dim.
    out = scaled_dot_product_attention_forward(
        query,
        key,
        value,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )

    # The Kunlunxin forward does not expose the online-softmax accumulator, so
    # there is no LSE to return; keep the shape the caller expects.
    lse = torch.zeros(
        query.shape[0],
        query.shape[1],
        query.shape[2],
        device=query.device,
        dtype=torch.float32,
    )

    # PyTorch < 2.9 returns [B, H, S_q], >= 2.9 returns [B, H, S_q, 1].
    torch_version = version.parse(torch.__version__.split("+")[0])
    if torch_version.release[:2] >= (2, 9):
        lse = lse.unsqueeze(-1)

    zero_seed = torch.zeros([], device=query.device, dtype=torch.int64)
    zero_offset = torch.zeros([], device=query.device, dtype=torch.int64)
    debug_mask = torch.zeros([0], device=query.device, dtype=query.dtype)

    return (
        out,
        lse,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        zero_seed,
        zero_offset,
        debug_mask,
    )


def _scaled_dot_product_cudnn_attention(
    query,
    key,
    value,
    attn_bias=None,
    compute_log_sumexp=True,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    """Scaled dot product attention with the same interface as ATen's op."""
    logger.debug("GEMS_KUNLUNXIN _SCALED_DOT_PRODUCT_CUDNN_ATTENTION")

    # Single tensor return on this backend (see the note in
    # ``cudnn_attention_forward`` above).
    output = scaled_dot_product_attention_forward(
        query,
        key,
        value,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=False,
    )

    if compute_log_sumexp:
        logsumexp = torch.zeros(
            query.shape[0],
            query.shape[1],
            query.shape[2],
            device=query.device,
            dtype=torch.float32,
        )
    else:
        logsumexp = torch.empty(0, device=query.device, dtype=torch.float32)

    cum_seq_q = torch.empty(0, device=query.device, dtype=torch.int32)
    cum_seq_k = torch.empty(0, device=query.device, dtype=torch.int32)
    max_q = query.shape[2]
    max_k = key.shape[2]
    philox_seed = torch.empty((), device=query.device, dtype=torch.int64)
    philox_offset = torch.empty((), device=query.device, dtype=torch.int64)

    if return_debug_mask:
        debug_attn_mask = torch.zeros(
            (
                query.shape[0],
                query.shape[1],
                query.shape[2],
                key.shape[2],
            ),
            device=query.device,
            dtype=query.dtype,
        )
    else:
        debug_attn_mask = torch.empty(0, device=query.device, dtype=query.dtype)

    return (
        output,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        philox_seed,
        philox_offset,
        debug_attn_mask,
    )
