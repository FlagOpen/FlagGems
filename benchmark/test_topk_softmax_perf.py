import pytest
import torch

import flag_gems.fused as fused

from .performance_utils import Benchmark

try:
    from vllm._custom_ops import topk_softmax as vllm_topk_softmax

    HAS_VLLM = True
except (ImportError, AttributeError) as e:
    print(f"Warning: Cannot import vLLM topk_softmax: {e}")
    HAS_VLLM = False
    vllm_topk_softmax = None


class TopKSoftmaxBenchmark(Benchmark):
    """
    Benchmark for comparing topk_softmax between vLLM (CUDA kernel) and FlagGems (Triton kernel).
    """

    def set_shapes(self, shape_file_path=None):
        # (num_tokens, num_experts, topk)
        topk_softmax_configs = [
            # small-scale
            (1, 8, 2),
            (1, 16, 2),
            (4, 16, 2),
            (4, 32, 2),
            (8, 16, 4),
            (8, 32, 4),
            (16, 64, 4),
            # medium-scale
            (32, 64, 8),
            (64, 64, 8),
            (128, 128, 16),
            (256, 128, 16),
            (512, 256, 16),
            (1024, 256, 32),
            # large-scale
            (4096, 64, 8),
            (8192, 128, 8),
            (16384, 256, 8),
        ]
        self.shapes = topk_softmax_configs

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self.topk_softmax_input_fn(config, cur_dtype, self.device)

    def topk_softmax_input_fn(self, config, dtype, device):
        """
        config: (num_tokens, num_experts, topk)
        """
        num_tokens, num_experts, k = config

        gating_output = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )

        topk_weights = torch.empty(num_tokens, k, device=device, dtype=torch.float32)
        topk_indices = torch.empty(num_tokens, k, device=device, dtype=torch.int32)
        token_expert_indices = torch.empty(
            num_tokens, k, device=device, dtype=torch.int32
        )

        yield (
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
        )


@pytest.mark.skipif(
    not HAS_VLLM, reason="vLLM not installed or topk_softmax unavailable"
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for topk_softmax benchmark",
)
@pytest.mark.perf
def test_perf_topk_softmax():
    bench = TopKSoftmaxBenchmark(
        op_name="topk_softmax",
        torch_op=vllm_topk_softmax,
        dtypes=[torch.float32],
    )

    bench.set_gems(fused.topk_softmax)

    bench.run()
