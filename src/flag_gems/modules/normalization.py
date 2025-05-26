# LayerNorm-related implementation.
# References:
# - PyTorch: https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/normalization.py#L321
# - vLLM: https://github.com/vllm-project/vllm/blob/v0.8.5/vllm/model_executor/layers/layernorm.py#L82
# - TransformerEngine:
#   https://github.com/NVIDIA/TransformerEngine/blob/v2.2.1/transformer_engine/pytorch/module/rmsnorm.py#L16
#
# Design notes:
# - Aligns with PyTorch’s RMSNorm interface for compatibility.
# - Works with or without flag_gems C++ wrappers.
# - Avoids relying on torch.nn.RMSNorm (introduced in v2.3.0) for broader compatibility.

import logging
import numbers
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Size
from torch.nn import Parameter, init

import flag_gems

logger = logging.getLogger(__name__)

try:
    from flag_gems import ext_ops  # noqa: F401

    has_c_extension = True
except ImportError:
    has_c_extension = False


__all__ = [
    "gems_rms_forward",
    "GemsRMSNorm",
]


def gems_rms_forward(
    x: torch.Tensor, residual: Optional[torch.Tensor], weight: torch.Tensor, eps: float
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    add_residual = residual is not None
    if add_residual:
        logger.debug("GEMS CUSTOM FUSED_ADD_RMS_NORM")
        if has_c_extension:
            torch.ops.flag_gems.fused_add_rms_norm(x, residual, weight, eps)
            return x, residual
        else:
            residual = x + residual
            return (
                flag_gems.rms_norm(residual, list(weight.size()), weight, eps),
                residual,
            )
    else:
        logger.debug("GEMS CUSTOM RMS_NORM")
        if has_c_extension:
            return torch.ops.flag_gems.rms_norm(x, weight, eps)
        else:
            return flag_gems.rms_norm(x, list(weight.size()), weight, eps)


class GemsRMSNorm(nn.Module):
    """
    Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    A custom implementation of RMSNorm that is compatible with both
    PyTorch RMSNorm and vLLM RMSNorm behavior.

    Unlike PyTorch's `torch.nn.RMSNorm`, which was introduced in v2.3.0,
    this implementation avoids version compatibility issues by directly
    inheriting from `torch.nn.Module`.

    The design mimics the interface and functionality of the official
    PyTorch and vLLM implementations, while providing additional flexibility
    to plug in custom fused CUDA/C++ kernels (via `flag_gems` extensions)
    when available.

    This module supports both standard RMS normalization and
    fused residual addition + RMS normalization.
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
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in __init__.
        """
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Runs forward pass.
        """
        return gems_rms_forward(x, residual, self.weight, self.eps)

    def extra_repr(self) -> str:
        """
        Extra information about the module.
        """
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )
