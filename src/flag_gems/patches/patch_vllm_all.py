from flag_gems.fused.reshape_and_cache import reshape_and_cache
from flag_gems.modules.normalization import gems_rms_forward
from flag_gems.patches.patch_util import patch_module_method


def custom_gems_rms_forward_cuda(self, x, residual=None):
    return gems_rms_forward(x, residual, self.weight, self.variance_epsilon)


def custom_gems_write_to_paged_cache(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    kv_cache_dtype,
    k_scale,
    v_scale,
):
    reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping.flatten(),
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def apply_gems_patches_to_vllm(verbose=True):
    from vllm.attention.ops.paged_attn import PagedAttention
    from vllm.model_executor.layers.layernorm import RMSNorm

    patch_module_method(RMSNorm, "forward_cuda", custom_gems_rms_forward_cuda, verbose)
    patch_module_method(
        PagedAttention,
        "write_to_paged_cache",
        custom_gems_write_to_paged_cache,
        verbose,
    )
