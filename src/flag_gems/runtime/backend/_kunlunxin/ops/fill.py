import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle


@libentry()
@triton.jit(do_not_specialize=["value_scalar"])
def fill_scalar_kernel(
    out_ptr,
    N,
    value_scalar,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid * BLOCK_SIZE + cols
    tl.store(out_ptr + offset, value_scalar, mask=offset < N)


@libentry()
@triton.jit
def fill_tensor_kernel(
    out_ptr,
    N,
    value_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid * BLOCK_SIZE + cols
    value_scalar = tl.load(value_ptr)  # load the value from the tensor.
    tl.store(out_ptr + offset, value_scalar, mask=offset < N)


def fill_tensor(input, value):
    logging.debug("GEMS FILL")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    N = out.numel()
    grid = 12
    BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(N, grid))

    with torch_device_fn.device(input.device):
        fill_tensor_kernel[grid,](out, N, value, BLOCK_SIZE)
    return out


def fill_scalar(input, value):
    logging.debug("GEMS FILL")
    out = torch.empty_like(input)
    N = out.numel()
    grid = 12
    BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(N, grid))

    with torch_device_fn.device(input.device):
        fill_scalar_kernel[grid,](out, N, value, BLOCK_SIZE)
    return out


def fill_tensor_(self, value):
    logging.debug("GEMS FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    N = self.numel()
    grid = 12
    BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(N, grid))

    with torch_device_fn.device(self.device):
        fill_tensor_kernel[grid,](self, N, value, BLOCK_SIZE)
    return self


def fill_scalar_(self, value):
    logging.debug("GEMS FILL_SCALAR_")
    N = self.numel()
    grid = 12
    BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(N, grid))

    with torch_device_fn.device(self.device):
        fill_scalar_kernel[grid,](self, N, value, BLOCK_SIZE)
    return self
