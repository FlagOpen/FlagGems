import itertools

import pytest
import torch

from .attri_util import (
    DEFAULT_BATCH,
    DEFAULT_NON_BLAS_BENCH_SHAPES,
    FLOAT_DTYPES,
    BenchLevel,
)
from .conftest import Config
from .performance_utils import Benchmark, unary_arg

CONSTRUCTOR_SHAPES = DEFAULT_NON_BLAS_BENCH_SHAPES[:]
if Config.bench_level == BenchLevel.COMPREHENSIVE:
    MORE_SHAPES = [(320, 15), (128, 64, 60)]
    MORE_BATCHS = [4, 20, 32]
    combinations = [
        (batch, *shape) for batch, shape in itertools.product(MORE_BATCHS, MORE_SHAPES)
    ]
    CONSTRUCTOR_SHAPES.extend(combinations)
    # add 1D shapes and 5D shapes
    CONSTRUCTOR_SHAPES.extend([(1,), (5,), (32, 5, 4, 7, 8)])


@pytest.mark.rand
def test_perf_rand():
    def rand_kwargs(dtype, batch, shape):
        return {"size": shape, "dtype": dtype, "device": "cuda"}

    bench = Benchmark(
        op_name="rand",
        torch_op=torch.rand,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=CONSTRUCTOR_SHAPES,
        kwargs_func=rand_kwargs,
    )
    bench.run()


@pytest.mark.randn
def test_perf_randn():
    def randn_kwargs(dtype, batch, shape):
        return {"size": shape, "dtype": dtype, "device": "cuda"}

    bench = Benchmark(
        op_name="randn",
        torch_op=torch.randn,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=CONSTRUCTOR_SHAPES,
        kwargs_func=randn_kwargs,
    )
    bench.run()


@pytest.mark.rand_like
def test_perf_rand_like():
    bench = Benchmark(
        op_name="rand_like",
        torch_op=torch.rand_like,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=CONSTRUCTOR_SHAPES,
    )
    bench.run()


@pytest.mark.randn_like
def test_perf_randn_like():
    bench = Benchmark(
        op_name="randn_like",
        torch_op=torch.randn_like,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=CONSTRUCTOR_SHAPES,
    )
    bench.run()


@pytest.mark.ones
def test_perf_ones():
    def ones_kwargs(dtype, batch, shape):
        return {"size": shape, "dtype": dtype, "device": "cuda"}

    bench = Benchmark(
        op_name="ones",
        torch_op=torch.ones,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=CONSTRUCTOR_SHAPES,
        kwargs_func=ones_kwargs,
    )
    bench.run()


@pytest.mark.zeros
def test_perf_zeros():
    def zeros_kwargs(dtype, batch, shape):
        return {"size": shape, "dtype": dtype, "device": "cuda"}

    bench = Benchmark(
        op_name="zeros",
        torch_op=torch.zeros,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=CONSTRUCTOR_SHAPES,
        kwargs_func=zeros_kwargs,
    )
    bench.run()


@pytest.mark.full
def test_perf_full():
    def full_kwargs(dtype, batch, shape):
        return {
            "size": shape,
            "fill_value": 3.1415926,
            "dtype": dtype,
            "device": "cuda",
        }

    bench = Benchmark(
        op_name="full",
        torch_op=torch.full,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=CONSTRUCTOR_SHAPES,
        kwargs_func=full_kwargs,
    )
    bench.run()


@pytest.mark.ones_like
def test_perf_ones_like():
    bench = Benchmark(
        op_name="ones_like",
        torch_op=torch.ones_like,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=CONSTRUCTOR_SHAPES,
    )
    bench.run()


@pytest.mark.zeros_like
def test_perf_zeros_like():
    bench = Benchmark(
        op_name="zeros_like",
        torch_op=torch.zeros_like,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=CONSTRUCTOR_SHAPES,
    )
    bench.run()


@pytest.mark.full_like
def test_perf_full_like():
    def full_kwargs(dtype, batch, shape):
        return {
            "input": torch.randn(shape, dtype=dtype, device="cuda"),
            "fill_value": 3.1415926,
        }

    bench = Benchmark(
        op_name="full_like",
        torch_op=torch.full_like,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=CONSTRUCTOR_SHAPES,
        kwargs_func=full_kwargs,
    )
    bench.run()
