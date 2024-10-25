import logging

import torch
import triton

from ..utils import pointwise_dynamic
from ..utils.random_utils import philox_cuda_seed_offset
from ..utils.shape_utils import broadcast_shapes, volume
from .randn import randn_kernel

UNROLL = 4


@pointwise_dynamic(
    is_tensor=[True, True, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_tensor_tensor(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, False, True], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def transform_func_tensor_float(val, std, mean):
    return val * std + mean


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
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


def normal_distribution(shape, device, *, generator=None):
    out = torch.empty(shape, device=device, dtype=torch.float32)
    N = volume(shape)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_cuda_seed_offset(increment)
    with torch.cuda.device(device):
        randn_kernel[grid_fn](out, N, philox_seed, philox_offset)
    return out


def normal_tensor_tensor(mean, std, *, generator=None):
    logging.debug("GEMS NORMAL_TENSOR_TENSOR")
    shape = broadcast_shapes([mean.shape, std.shape])
    device = mean.device
    out = normal_distribution(shape, device)
    return transform_func_tensor_tensor(out, std, mean)


def normal_tensor_float(mean, std, *, generator=None):
    logging.debug("GEMS NORMAL_TENSOR_FLOAT")
    shape = mean.shape
    device = mean.device
    out = normal_distribution(shape, device)
    return transform_func_tensor_float(out, std, mean)


def normal_float_tensor(mean, std, *, generator=None):
    logging.debug("GEMS NORMAL_FLOAT_TENSOR")
    shape = std.shape
    device = std.device
    out = normal_distribution(shape, device)
    return transform_func_float_tensor(out, std, mean)
