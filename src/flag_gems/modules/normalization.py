# LayerNorm-related implementation.
# References:
# - PyTorch: https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/normalization.py#L321
# - vLLM: https://github.com/vllm-project/vllm/blob/v0.8.5/vllm/model_executor/layers/layernorm.py#L82
# - TransformerEngine:
# https://github.com/NVIDIA/TransformerEngine/blob/v2.2.1/transformer_engine/pytorch/module/rmsnorm.py#L16
#
# Our design:
# - Aligns with PyTorch's interface for compatibility and ease of integration.

from typing import List, Optional, Union

import torch
from torch import Size
from torch.nn import RMSNorm

__all__ = [
    "GemsRMSNorm",
]


class GemsRMSNorm(RMSNorm):
    """
    Applies Root Mean Square Layer Normalization over a mini-batch of inputs.
    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Union[int, list[int], Size]
    eps: Optional[float]
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: List[int],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Runs forward pass.
        """
        add_residual = residual is not None
        if add_residual:
            return torch.ops.flag_gems.fused_add_rms_norm(
                x, residual, self.weight, self.eps
            )
        else:
            return torch.ops.flag_gems.rms_norm(x, self.weight, self.eps)
