import logging

import triton

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def lerp_tensor_kernel(input, end, weight):
    return input + weight * (end - input)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def lerp_scalar_kernel(input, end, weight):
    return input + weight * (end - input)


def lerp_tensor(input, end, weight):
    logging.debug("GEMS LERP TENSOR")
    out = lerp_tensor_kernel(input, end, weight)
    return out


def lerp_tensor_(input, end, weight):
    logging.debug("GEMS LERP INPLACE TENSOR")
    return lerp_tensor_kernel(input, end, weight, out0=input)


def lerp_scalar(input, end, weight):
    logging.debug("GEMS LERP TENSOR")
    out = lerp_scalar_kernel(input, end, weight)
    return out


def lerp_scalar_(input, end, weight):
    logging.debug("GEMS LERP INPLACE TENSOR")
    return lerp_scalar_kernel(input, end, weight, out0=input)
