import pytest
import torch

import flag_gems

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    BOOL_TYPES,
    DISTRIBUTION_SHAPES,
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_equal,
)
from .conftest import TO_CPU

device = flag_gems.device


@pytest.mark.rand
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.rand(shape, dtype=dtype, device=device)
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()


@pytest.mark.randn
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_randn(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.randn(shape, dtype=dtype, device=device)
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01


@pytest.mark.rand_like
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    with flag_gems.use_gems():
        res_out = torch.rand_like(x)
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()


@pytest.mark.randn_like
@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_randn_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    with flag_gems.use_gems():
        res_out = torch.randn_like(x)
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01


@pytest.mark.zeros
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
def test_accuracy_zeros(shape, dtype):
    # without dtype
    with flag_gems.use_gems():
        res_out = torch.zeros(shape, device=flag_gems.device)
    gems_assert_equal(res_out, torch.zeros(shape, device="cpu" if TO_CPU else device))

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    gems_assert_equal(
        res_out, torch.zeros(shape, dtype=dtype, device="cpu" if TO_CPU else device)
    )


@pytest.mark.ones
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
def test_accuracy_ones(shape, dtype):
    # without dtype
    with flag_gems.use_gems():
        res_out = torch.ones(shape, device=flag_gems.device)
    gems_assert_equal(res_out, torch.ones(shape, device="cpu" if TO_CPU else device))

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    gems_assert_equal(
        res_out, torch.ones(shape, dtype=dtype, device="cpu" if TO_CPU else device)
    )


@pytest.mark.full
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("fill_value", [3.1415926, 2, False])
def test_accuracy_full(shape, dtype, fill_value):
    # without dtype
    ref_out = torch.full(shape, fill_value, device="cpu" if TO_CPU else device)
    with flag_gems.use_gems():
        res_out = torch.full(shape, fill_value, device=flag_gems.device)
    gems_assert_equal(res_out, ref_out)

    # with dtype
    ref_out = torch.full(
        shape, fill_value, dtype=dtype, device="cpu" if TO_CPU else device
    )
    with flag_gems.use_gems():
        res_out = torch.full(shape, fill_value, dtype=dtype, device=flag_gems.device)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.zeros_like
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_zeros_like(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device="cpu" if TO_CPU else device)
    with flag_gems.use_gems():
        res_out = torch.zeros_like(x)
    gems_assert_equal(res_out, torch.zeros_like(x))


@pytest.mark.ones_like
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ones_like(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device="cpu" if TO_CPU else device)
    with flag_gems.use_gems():
        res_out = torch.ones_like(x)
    gems_assert_equal(res_out, torch.ones_like(x))


@pytest.mark.full_like
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("xdtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("fill_value", [3.1415926, 2, False])
def test_accuracy_full_like(shape, dtype, xdtype, fill_value):
    x = torch.empty(size=shape, dtype=xdtype, device="cpu" if TO_CPU else device)

    # without dtype
    with flag_gems.use_gems():
        res_out = torch.full_like(x, fill_value)
    gems_assert_equal(res_out, torch.full_like(x, fill_value))

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.full_like(x, fill_value, dtype=dtype)
    gems_assert_equal(res_out, torch.full_like(x, fill_value, dtype=dtype))


@pytest.mark.randperm
@pytest.mark.parametrize("n", [123, 12345, 123456])
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES)
def test_accuracy_randperm(n, dtype):
    if n > torch.iinfo(torch.int16).max and dtype == torch.int16:
        return

    ref_out = torch.randperm(n, dtype=dtype, device="cpu" if TO_CPU else device)
    with flag_gems.use_gems():
        res_out = torch.randperm(n, dtype=dtype, device=flag_gems.device)
    sorted_ref, _ = torch.sort(ref_out)
    sorted_res, _ = torch.sort(res_out)
    gems_assert_equal(sorted_res, sorted_ref)
