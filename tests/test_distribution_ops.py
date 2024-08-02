import numpy as np
import pytest
import scipy
import torch

import flag_gems

from .accuracy_utils import DISTRIBUTION_SHAPES, FLOAT_DTYPES


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_rand(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.rand(shape, dtype=dtype, device="cuda")
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()
    pvalue = scipy.stats.kstest(
        res_out.cpu().numpy().flatten(), lambda x: scipy.stats.uniform.cdf(x)
    ).pvalue
    assert pvalue > 0.05


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_randn(shape, dtype):
    with flag_gems.use_gems():
        res_out = torch.randn(shape, dtype=dtype, device="cuda")
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01
    pvalue = scipy.stats.kstest(
        res_out.cpu().numpy().flatten(), lambda x: scipy.stats.norm.cdf(x)
    ).pvalue
    assert pvalue > 0.05


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        res_out = torch.rand_like(x)
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
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
    pvalue = scipy.stats.kstest(
        res_out.cpu().numpy().flatten(),
        lambda x: scipy.stats.norm.cdf(x, loc=3.0, scale=10.0),
    ).pvalue
    assert pvalue > 0.05


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_uniform(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        x.uniform_(-3, 3)
    assert (x <= 3.0).all()
    assert (x >= -3.0).all()
    pvalue = scipy.stats.kstest(
        x.cpu().numpy().flatten(),
        lambda x: scipy.stats.uniform.cdf(x, loc=-3.0, scale=6.0),
    ).pvalue
    assert pvalue > 0.05


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exponential_(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        x.exponential_()
    assert x.min() > 0


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("lambd", (0.01, 0.5, 100.0))
def test_accuracy_exponential_pvalue(shape, dtype, lambd):
    x = torch.empty(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        x.exponential_(lambd=lambd)
    expo_cdf = lambda x: np.where(x < 0, 0, 1.0 - np.exp(-lambd * x))
    pvalue = scipy.stats.kstest(x.cpu().numpy().flatten(), expo_cdf).pvalue
    assert pvalue > 0.05
