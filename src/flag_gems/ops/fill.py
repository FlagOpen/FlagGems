import logging

import triton

from ..runtime import torch_device_fn
from ..utils import pointwise_dynamic


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_scalar_kernel(out, value_scalar):
    return value_scalar


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_tensor_kernel(out, value):
    return value


def fill_scalar(input, value):
    logging.debug("GEMS FILL (Dynamic)")
    with torch_device_fn.device(input.device):
        return fill_scalar_kernel(input, value)


def fill_tensor(input, value):
    logging.debug("GEMS FILL (Dynamic)")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    with torch_device_fn.device(input.device):
        return fill_tensor_kernel(input, value)


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
