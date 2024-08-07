import numpy as np
import pytest
import scipy
import torch

import flag_gems

from .accuracy_utils import DISTRIBUTION_SHAPES, FLOAT_DTYPES


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


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_uniform(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        x.uniform_(-3, 3)
    assert (x <= 3.0).all()
    assert (x >= -3.0).all()


@pytest.mark.parametrize("shape", DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exponential_(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device="cuda")
    with flag_gems.use_gems():
        x.exponential_()
    assert x.min() > 0


@pytest.mark.parametrize("shape", [(1000,), (100, 1000)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("n_samples", [2048])
def test_accuracy_multinomial_with_replacement(shape, dtype, n_samples):
    dist = torch.zeros(size=shape, dtype=dtype, device="cuda")
    Index = [5, 13, 42]
    dist[..., Index] = 1
    with flag_gems.use_gems():
        res_out = torch.multinomial(dist, n_samples, True)
    index, tally = torch.unique(res_out, sorted=True, return_counts=True)
    assert index.tolist() == Index
    # Do a simple Chi-square test
    tally = np.array(tally.tolist())
    expected = tally
    expected[:] = tally.mean()
    observed = tally
    chi2, pvalue = scipy.stats.chisquare(observed, expected)
    assert pvalue > 0.05


@pytest.mark.parametrize("pool", [100])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("n_samples", [10])
def test_accuracy_multinomial_without_replacement(pool, dtype, n_samples):
    n_draws = 1000
    dist = torch.zeros(size=(pool,), dtype=dtype, device="cuda").broadcast_to(
        n_draws, pool
    )
    indices = torch.randint(0, pool, (50,), device="cuda").unique()
    dist[:, indices] = 1
    with flag_gems.use_gems():
        res_out = torch.multinomial(dist, n_samples, False)
    # Verifies uniqueness
    for draw in range(n_draws):
        assert res_out[draw].unique().size(0) == res_out.size(1)
    # Chi-square tests
    samples, count = res_out.unique(return_counts=True)
    dist = dist[0][samples]
    dist = dist / dist.sum()
    # The expected number of samples must equal the observed number of samples exactly
    observed_samples = n_samples * n_draws
    expected_count = torch.round(dist * n_samples * n_draws)
    expected_count[0] += observed_samples - expected_count.sum()
    chi2, pvalue = scipy.stats.chisquare(count.tolist(), expected_count.tolist())
    assert pvalue > 0.05
