import torch

from flag_gems.fused import flash_mla
from flag_gems.modules.normalization import gems_rms_forward
from flag_gems.patches.patch_util import patch_module_method


def custom_gems_rms_forward_cuda(self, x, residual=None):
    return gems_rms_forward(x, residual, self.weight, self.variance_epsilon)


def custom_gems_flash_mla_forward(
    self,
    q_nope,
    q_pe,
    kv_c_and_k_pe_cache,
    attn_metadata,
) -> torch.Tensor:
    assert kv_c_and_k_pe_cache.numel() > 0
    assert attn_metadata.decode is not None

    if self.kv_cache_dtype.startswith("fp8"):
        raise NotImplementedError("FP8 Triton MLA not yet supported")

    batch, num_head_q, head_dim_v = q_nope.shape
    seqlen_q = 1

    q = torch.cat([q_nope, q_pe], dim=-1)
    head_dim = q.shape[-1]
    q = q.view(batch, seqlen_q, num_head_q, head_dim)

    # Add a head dim of 1
    kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
    PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

    block_table = attn_metadata.decode.block_table
    output = flash_mla(
        q,
        block_table,
        kv_c_and_k_pe_cache,
        None,
        PAGE_SIZE,
        batch,
        seqlen_q,
        attn_metadata.decode.seq_lens,
        num_head_q,
        None,
        head_dim,
        head_dim_v,
        True,
    )

    o = self._v_up_proj_and_o_proj(output)
    return o


def apply_gems_patches_to_vllm(verbose=True):
    from vllm.model_executor.layers.layernorm import RMSNorm

    patch_module_method(RMSNorm, "forward_cuda", custom_gems_rms_forward_cuda, verbose)

    from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl

    patch_module_method(
        TritonMLAImpl, "_forward_decode", custom_gems_flash_mla_forward, verbose
    )
