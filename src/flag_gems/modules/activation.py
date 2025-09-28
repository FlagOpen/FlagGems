import logging

import torch
import torch.nn as nn

import flag_gems
from flag_gems.config import use_c_extension

logger = logging.getLogger(__name__)


def _c_extension_available() -> bool:
    return bool(
        use_c_extension
        and hasattr(torch.ops, "flag_gems")
        and hasattr(torch.ops.flag_gems, "silu_and_mul")
        and hasattr(torch.ops.flag_gems, "silu_and_mul_out")
    )


__all__ = [
    "gems_silu_and_mul",
    "GemsSiluAndMul",
]


def gems_silu_and_mul(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    if _c_extension_available():
        logger.debug("GEMS CUSTOM SILU_AND_MUL FORWARD(C EXTENSION)")
        return torch.ops.flag_gems.silu_and_mul(x, y)
    logger.debug("GEMS CUSTOM SILU_AND_MUL FORWARD")
    return flag_gems.silu_and_mul(x, y)


class GemsSiluAndMul(nn.Module):
    """
    Fused Silu and Mul activation function.
    The function computes torch.mul(torch.nn.functional.silu(x), y) in a fused way.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return gems_silu_and_mul(x, y)
