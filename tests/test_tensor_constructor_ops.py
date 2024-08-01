import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, POINTWISE_SHAPES, gems_assert_equal


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_zeros(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.zeros(shape, dtype=dtype, device="cuda")
    gems_assert_equal(res_out, torch.zeros(shape, dtype=dtype, device="cuda"))


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ones(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.ones(shape, dtype=dtype, device="cuda")
    gems_assert_equal(res_out, torch.ones(shape, dtype=dtype, device="cuda"))


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_full(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.full(shape, 3.1415926, dtype=dtype, device="cuda")
    gems_assert_equal(res_out, torch.full(shape, 3.1415926, dtype=dtype, device="cuda"))


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_zeros_like(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        res_out = torch.zeros_like(x)
    gems_assert_equal(res_out, torch.zeros_like(x))


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ones_like(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        res_out = torch.ones_like(x)
    gems_assert_equal(res_out, torch.ones_like(x))


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_full_like(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        res_out = torch.full_like(x, 3.1415926)
    gems_assert_equal(res_out, torch.full_like(x, 3.1415926))
