import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, POINTWISE_SHAPES


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.rand(shape, dtype=dtype, device="cuda")
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_randn(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.randn(shape, dtype=dtype, device="cuda")
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        res_out = torch.rand_like(x)
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_normal(shape, dtype):
    loc = torch.full(size=shape, fill_value=3.0, dtype=dtype, device="cuda")
    scale = torch.full(size=shape, fill_value=10.0, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        res_out = torch.distributions.normal.Normal(loc, scale).sample()
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean - 3.0) < 0.1
    assert torch.abs(std - 10.0) < 0.1


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_uniform(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        x.uniform_(-3, 3)
    assert (x <= 3.0).all()
    assert (x >= -3.0).all()
