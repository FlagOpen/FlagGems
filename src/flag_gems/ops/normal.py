import logging

import torch
import triton

import flag_gems.ops.randn as randn
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.random_utils import philox_cuda_seed_offset
from flag_gems.utils.shape_utils import broadcast_shapes, volume

UNROLL = 4


@pointwise_dynamic(
    is_tensor=[True, True, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_tensor_tensor(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_tensor_float(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, False, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_float_tensor(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_float_float(val, std, mean):
    return val * std + mean


def normal(mean, std, *, generator=None):
    logging.debug("GEMS NORMAL")
    shape = broadcast_shapes([mean.shape, std.shape])
    out = torch.empty(shape, device=mean.device, dtype=torch.float32)
    N = volume(shape)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_cuda_seed_offset(increment)
    with torch.cuda.device(mean.device):
        randn.randn_kernel[grid_fn](out, N, philox_seed, philox_offset)
    if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
        return transform_func_tensor_tensor(out, std, mean)
    elif isinstance(mean, torch.Tensor):
        return transform_func_tensor_float(out, std, mean)
    elif isinstance(std, torch.Tensor):
        return transform_func_float_tensor(out, std, mean)
    else:
        return transform_func_float_float(out, std, mean)
