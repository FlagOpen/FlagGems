import itertools

import pytest
import torch

from .attri_util import (
    DEFAULT_BATCH,
    DEFAULT_NON_BLAS_BENCH_SHAPES,
    FLOAT_DTYPES,
    LEGACY_SHAPES,
    POINTWISE_BATCH,
    BenchLevel,
)
from .performance_utils import Benchmark, Config, unary_arg

DISTRIBUTION_SHAPES = DEFAULT_NON_BLAS_BENCH_SHAPES[:]
if Config.bench_level == BenchLevel.COMPREHENSIVE:
    MORE_SHAPES = [(320, 15), (128, 64, 60)]
    MORE_BATCHS = [4, 20, 32]
    combinations = [
        (batch, *shape) for batch, shape in itertools.product(MORE_BATCHS, MORE_SHAPES)
    ]
    DISTRIBUTION_SHAPES.extend(combinations)


@pytest.mark.normal
def test_perf_normal():
    def normal_arg(dtype, batch, size):
        loc = torch.full(size=(size, batch), fill_value=3.0, dtype=dtype, device="cuda")
        scale = torch.full(
            size=(size, batch), fill_value=10.0, dtype=dtype, device="cuda"
        )
        return loc, scale

    bench = Benchmark(
        op_name="normal",
        torch_op=torch.distributions.normal.Normal,
        arg_func=normal_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=LEGACY_SHAPES,
    )
    bench.run()


@pytest.mark.uniform_
def test_perf_uniform():
    bench = Benchmark(
        op_name="uniform_",
        torch_op=torch.Tensor.uniform_,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=DISTRIBUTION_SHAPES,
    )
    bench.run()


@pytest.mark.exponential_
def test_perf_exponential_():
    bench = Benchmark(
        op_name="exponential_",
        torch_op=torch.Tensor.exponential_,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=LEGACY_SHAPES,
    )
    bench.run()
