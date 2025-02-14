import logging

import torch
import triton
import triton.language as tl

from ..runtime.moduel_tool import tl_extra_module
from ..utils import pointwise_dynamic

# _isfinited = tl_extra_module.isfinited
# _finitef = tl_extra_module.finitef


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isfinite_func(x):
    if x.dtype.is_fp64():
        return (x.to(tl.int64, bitcast = True) & 0x7fffffffffffffff) < 0x7ff0000000000000
    elif x.dtype.is_fp32():
        return (x.to(tl.int32, bitcast = True) & 0x7fffffff) < 0x7f800000
    elif x.dtype.is_fp16():
        return (x.to(tl.int16, bitcast = True) & 0x7fff) < 0x7c00
    elif x.dtype.is_bf16():
        return (x.to(tl.int16, bitcast = True) & 0x7fff) < 0x7f80

    # return _isfinited(x) if x.dtype.is_fp64() else _finitef(x.to(tl.float32))


def isfinite(
    A: torch.Tensor,
) -> torch.Tensor:
    logging.debug("GEMS ISFINITE")
    if A.is_floating_point():
        legal_dtype = [torch.float32, torch.float16, torch.bfloat16]
        assert A.dtype in legal_dtype, f"isfinite input float dtype should in {str(legal_dtype)}, get {str(A.dtype)}"
        return isfinite_func(A)
    else:
        return torch.full(A.shape, True, dtype=torch.bool, device=A.device)
