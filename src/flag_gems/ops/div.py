import logging

import torch
import triton

from ..utils import unwrap


@triton.jit
def true_div_func(x, y):
    return x / y


@triton.jit
def true_div_func_tensor_scalar(x, y):
    return (x / y).to(x.type.element_ty)


@triton.jit
def true_div_func_scalar_tensor(x, y):
    return (x / y).to(y.type.element_ty)


def true_divide(A, B):
    logging.debug("GEMS TRUE_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return unwrap(true_div_func[(1,)](A, B))
    elif isinstance(A, torch.Tensor):
        return unwrap(true_div_func_tensor_scalar[(1,)](A, B))
    elif isinstance(B, torch.Tensor):
        return unwrap(true_div_func_scalar_tensor[(1,)](A, B))
    else:
        # Both scalar
        return A / B


@triton.jit
def trunc_div_func(x, y):
    return triton.div_rz(x, y)


@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return triton.div_rz(x, y)


@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return triton.div_rz(x, y)


def trunc_divide(A, B):
    logging.debug("GEMS TRUNC_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return unwrap(trunc_div_func[(1,)](A, B))
    elif isinstance(A, torch.Tensor):
        return unwrap(trunc_div_func_tensor_scalar[(1,)](A, B))
    elif isinstance(B, torch.Tensor):
        return unwrap(trunc_div_func_scalar_tensor[(1,)](A, B))
    else:
        # Both scalar
        return A / B


@triton.jit
def floor_div_func(x, y):
    return x // y


@triton.jit
def floor_div_func_tensor_scalar(x, y):
    return x // y


@triton.jit
def floor_div_func_scalar_tensor(x, y):
    return x // y


def floor_divide(A, B):
    logging.debug("GEMS FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return unwrap(floor_div_func[(1,)](A, B))
    elif isinstance(A, torch.Tensor):
        return unwrap(floor_div_func_tensor_scalar[(1,)](A, B))
    elif isinstance(B, torch.Tensor):
        return unwrap(floor_div_func_scalar_tensor[(1,)](A, B))
    else:
        # Both scalar
        return A // B


def div(A, B, rounding_mode=None):
    if rounding_mode is None:
        return true_divide(A, B)
    elif rounding_mode == "trunc":
        return trunc_divide(A, B)
    elif rounding_mode == "floor":
        return floor_divide(A, B)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)
