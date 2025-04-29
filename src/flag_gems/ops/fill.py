import logging

import torch
import triton

from ..runtime import torch_device_fn
from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def fill_scalar_kernel(
    out,
    value_scalar,
):
    out = value_scalar
    return out


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def fill_tensor_kernel(
    out,
    value,
):
    out = value
    return out


def fill_tensor(input, value):
    logging.debug("GEMS FILL")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        result = fill_tensor_kernel(out, value)
    return result


def fill_scalar(input, value):
    logging.debug("GEMS FILL")
    out = torch.empty_like(input)
    with torch_device_fn.device(input.device):
        result = fill_scalar_kernel(out, value)
    return result


def fill_tensor_(self, value):
    logging.debug("GEMS FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    with torch_device_fn.device(self.device):
        fill_tensor_kernel(self, value, out0=self)
    return self


def fill_scalar_(self, value):
    logging.debug("GEMS FILL_SCALAR_")
    with torch_device_fn.device(self.device):
        fill_scalar_kernel(self, value, out0=self)
    return self
