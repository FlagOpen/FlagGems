import pytest
import torch

from flag_gems.fused.topk_softmax import topk_softmax


def topk_softmax_torch_reference(gating_output: torch.Tensor, topk: int):
    probs = torch.softmax(gating_output, dim=-1)
    topk_values, topk_indices = torch.topk(
        probs, k=topk, dim=-1, largest=True, sorted=True
    )
    num_tokens = gating_output.shape[0]
    source_rows = torch.arange(topk, device=gating_output.device).view(
        1, -1
    ) * num_tokens + torch.arange(num_tokens, device=gating_output.device).view(-1, 1)
    return topk_values, topk_indices, source_rows


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64, torch.uint32])
@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    [
        (1, 4, 2),
        (4, 8, 2),
        (8, 16, 4),
        (32, 64, 8),
        (128, 128, 16),
        (500, 255, 30),
        (512, 256, 32),
        (1024, 512, 32),
    ],
)
def test_topk_softmax(num_tokens, num_experts, topk, index_dtype):
    torch.manual_seed(42)
    device = "cuda"

    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )

    topk_weights = torch.empty((num_tokens, topk), device=device, dtype=torch.float32)
    topk_indices = torch.empty((num_tokens, topk), device=device, dtype=index_dtype)
    token_expert_indices = torch.empty(
        (num_tokens, topk), device=device, dtype=torch.int32
    )

    topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output)

    ref_weights, ref_indices, ref_source_rows = topk_softmax_torch_reference(
        gating_output, topk
    )

    assert topk_weights.shape == (num_tokens, topk)
    assert topk_indices.shape == (num_tokens, topk)
    assert token_expert_indices.shape == (num_tokens, topk)

    assert torch.allclose(topk_weights, ref_weights, atol=1e-5)
    assert torch.equal(topk_indices.cpu(), ref_indices.to(index_dtype).cpu())
    assert torch.equal(token_expert_indices.cpu(), ref_source_rows.cpu())
