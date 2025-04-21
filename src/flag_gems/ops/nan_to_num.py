import logging
import torch

import triton
import triton.language as tl

from ..utils import pointwise_dynamic, tl_extra_shim

_isnan = tl_extra_shim.isnan

@pointwise_dynamic(
    is_tensor=[True, False, False, False],
    promotion_methods=[(0, "DEFAULT")])
@triton.jit
def nan_to_num_func(x, nan, posinf, neginf):
    x_nan = _isnan(x.to(tl.float32))
    x_posinf = x == float('inf')
    x_neginf = x == -float('inf')
    x = tl.where(x_nan, nan, x)
    x = tl.where(x_posinf, posinf, x)
    x = tl.where(x_neginf, neginf, x)
    return x

def nan_to_num(A, nan=0.0, posinf=None, neginf=None):
    logging.debug("GEMS NAN_TO_NUM TENSOR")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = -torch.finfo(A.dtype).max
    return nan_to_num_func(A, nan, posinf, neginf)