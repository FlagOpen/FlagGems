import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True], output_dtypes=[torch.bool])
@triton.jit
def isfinite_func(x):
    cast_x = x if x.dtype == torch.float64 else x.to(tl.float32)
    return (cast_x == cast_x) & (cast_x != float("inf")) & (cast_x != float("-inf"))


def isfinite(
    A: torch.Tensor,
) -> torch.Tensor:
    logging.debug("GEMS ISFINITE")
    if A.dtype in (
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        return isfinite_func(A)
    else:
        return torch.full(A.shape, True, dtype=torch.bool, device=A.device)
