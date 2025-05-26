import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset
from flag_gems.utils.shape_utils import broadcast_shapes, volume

from ..utils.pointwise_dynamic import pointwise_dynamic
from .randn import choose_unroll, randn_kernel_1, randn_kernel_2

# UNROLL = 4

logger = logging.getLogger(__name__)


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
    # grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    cluster_num = 12
    UNROLL = choose_unroll(N)
    BLOCK_SIZE = min(triton.next_power_of_2(triton.cdiv(N, cluster_num * UNROLL)), 1024)
    grid_fn = triton.cdiv(N, BLOCK_SIZE * UNROLL)

    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(device):
        if UNROLL <= 4:
            randn_kernel_1[(grid_fn,)](
                out, N, philox_seed, philox_offset, BLOCK_SIZE, UNROLL
            )
        else:
            randn_kernel_2[(grid_fn,)](
                out, N, philox_seed, philox_offset, BLOCK_SIZE, UNROLL
            )
    return out


def normal_tensor_tensor(mean, std, *, generator=None):
    logger.debug("GEMS NORMAL_TENSOR_TENSOR")
    shape = broadcast_shapes([mean.shape, std.shape])
    device = mean.device
    out = normal_distribution(shape, device)
    return transform_func_tensor_tensor(out, std, mean)


def normal_tensor_float(mean, std, *, generator=None):
    logger.debug("GEMS NORMAL_TENSOR_FLOAT")
    shape = mean.shape
    device = mean.device
    out = normal_distribution(shape, device)
    return transform_func_tensor_float(out, std, mean)


def normal_float_tensor(mean, std, *, generator=None):
    logger.debug("GEMS NORMAL_FLOAT_TENSOR")
    shape = std.shape
    device = std.device
    out = normal_distribution(shape, device)
    return transform_func_float_tensor(out, std, mean)
