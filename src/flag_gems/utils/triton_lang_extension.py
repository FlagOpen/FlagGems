"""
Triton device functions.

Custom triton device functions that we need to use.

NOTE:
Do not try to add triton builtin-style functions(functions with an ir builder in its
arguments) here. We only define device-functions(triton.jit decorated functions with
return statement) here.

These functions can be used in kernel progamming and are not bound to any grid.
"""

import triton
from triton import language as tl

from flag_gems.utils.triton_lang_helper import use_tl_extra


@triton.jit
def program_id(
    axis: int,
) -> tl.tensor:
    return tl.program_id(axis).to(tl.int64)


@triton.jit
def num_programs(
    axis: int,
) -> tl.tensor:
    return tl.num_programs(axis).to(tl.int64)


@triton.jit
def promote_to_tensor(x):
    # Addition promotes to tensor for us
    return x + tl.zeros((1,), tl.int1)


@triton.jit
def is_floating(x):
    return promote_to_tensor(x).dtype.is_floating()


@triton.jit
def minimum_with_index_tie_break_right(a_value, a_index, b_value, b_index):
    mask = a_value < b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        # Consider NaNs as equal
        equal |= a_isnan and b_isnan

    # Prefer highest index if values are equal
    mask |= equal & (a_index > b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def maximum_with_index_tie_break_right(a_value, a_index, b_value, b_index):
    mask = a_value > b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        # Consider NaNs as equal
        equal |= a_isnan and b_isnan

    # Prefer highest index if values are equal
    mask |= equal & (a_index > b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@use_tl_extra
@triton.jit
def div_rn(x, y):
    """div_rn default - round to nearest"""
    result = x / y
    return tl.floor(result + 0.5)


@use_tl_extra
@triton.jit
def div_rz(x, y):
    """div_rz default - round toward zero"""
    result = x / y
    return tl.where(result >= 0, tl.floor(result), tl.ceil(result))


@use_tl_extra
@triton.jit
def fmod(x, y):
    """fmod default - floating point modulo"""
    quotient = div_rz(x, y)
    return x - y * quotient


@use_tl_extra
@triton.jit
def trunc(x):
    """trunc default - truncate to integer"""
    return tl.where(x >= 0, tl.floor(x), tl.ceil(x))


# --- Pointwise Functions ---

# src/flag_gems/ops/add.py


def add_func(x, y, alpha):
    return x + y * alpha


def add_func_tensor_scalar(x, y, alpha):
    return x + y * alpha


def add_func_scalar_tensor(x, y, alpha):
    return x + y * alpha


# src/flag_gems/ops/fill.py for lib/fill.cpp
def fill_scalar_func(inp, value_scalar):
    return tl.full(inp.shape, value_scalar, dtype=inp.dtype)


def fill_tensor_func(inp, value):
    return value


# zeros 还不是pointwise dynamic

# src/flag_gems/ops/copy.py for lib/cat.cpp


def copy(src):
    return src
