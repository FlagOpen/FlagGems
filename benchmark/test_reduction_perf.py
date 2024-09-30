import itertools

import pytest
import torch

from .attri_util import (
    DEFAULT_NON_BLAS_BENCH_SHAPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    REDUCTION_BATCH,
    BenchLevel,
)
from .performance_utils import Benchmark, Config, unary_arg

REDUCTION_SHAPES = DEFAULT_NON_BLAS_BENCH_SHAPES[:]
if Config.bench_level == BenchLevel.COMPREHENSIVE:
    MORE_SHAPES = [(320, 15), (128, 64, 60)]
    MORE_BATCHS = [4, 20, 32]
    combinations = [
        (batch, *shape) for batch, shape in itertools.product(MORE_BATCHS, MORE_SHAPES)
    ]
    REDUCTION_SHAPES.extend(combinations)


# TODO: Set the `keepdim` and `dim` parameters when the benchmark level is set to comprehensive.
@pytest.mark.all
def test_perf_all():
    bench = Benchmark(
        op_name="all",
        torch_op=torch.all,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


@pytest.mark.amax
def test_perf_amax():
    bench = Benchmark(
        op_name="amax",
        torch_op=torch.amax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


@pytest.mark.any
def test_perf_any():
    bench = Benchmark(
        op_name="any",
        torch_op=torch.any,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


# TODO: Set the `keepdim` and `dim` parameters when the benchmark level is set to comprehensive.
@pytest.mark.argmax
def test_perf_argmax():
    bench = Benchmark(
        op_name="argmax",
        torch_op=torch.argmax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


@pytest.mark.CrossEntropyLoss
def test_perf_cross_entropy_loss():
    def cross_entropy_loss_args(dtype, batch, shape):
        inp = torch.randn(size=shape, dtype=dtype, device="cuda")
        target = torch.randint(
            0,
            shape[-1],
            [
                shape[0],
            ],
            device="cuda",
        )
        return inp, target

    bench = Benchmark(
        op_name="CrossEntropyLoss",
        torch_op=torch.nn.CrossEntropyLoss(),
        arg_func=cross_entropy_loss_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


@pytest.mark.cumsum
def test_perf_cumsum():
    def cumsum_args(dtype, batch, shape):
        if dtype in INT_DTYPES:
            inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
        else:
            inp = torch.randn(shape, dtype=dtype, device="cuda")
        return inp, 1

    bench = Benchmark(
        op_name="cumsum",
        torch_op=torch.cumsum,
        arg_func=cumsum_args,
        dtypes=FLOAT_DTYPES + INT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


@pytest.mark.nonzero
def test_perf_nonzero():
    def nonzero_args(dtype, batch, shape):
        if dtype == torch.bool:
            inp = torch.randint(0, 2, shape, dtype=torch.int, device="cuda").to(dtype)
        elif dtype in INT_DTYPES:
            inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
        else:
            inp = torch.randn(shape, dtype=dtype, device="cuda")
        return (inp,)

    bench = Benchmark(
        op_name="nonzero",
        torch_op=torch.nonzero,
        arg_func=nonzero_args,
        dtypes=FLOAT_DTYPES + INT_DTYPES + [torch.bool],
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()

@pytest.mark.log_softmax
def test_perf_log_softmax():
    bench = Benchmark(
        op_name="log_softmax",
        torch_op=torch.nn.functional.log_softmax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


@pytest.mark.max
def test_perf_max():
    bench = Benchmark(
        op_name="max",
        torch_op=torch.max,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


@pytest.mark.mean
def test_perf_mean():
    bench = Benchmark(
        op_name="mean",
        torch_op=torch.mean,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


@pytest.mark.min
def test_perf_min():
    bench = Benchmark(
        op_name="min",
        torch_op=torch.min,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


@pytest.mark.prod
def test_perf_prod():
    bench = Benchmark(
        op_name="prod",
        torch_op=torch.prod,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


@pytest.mark.softmax
def test_perf_softmax():
    bench = Benchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


def test_perf_softmax_backward():
    bench = Benchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
        is_backward=True,
    )
    bench.run()


@pytest.mark.sum
def test_perf_sum():
    bench = Benchmark(
        op_name="sum",
        torch_op=torch.sum,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


@pytest.mark.var_mean
def test_perf_var_mean():
    bench = Benchmark(
        op_name="var_mean",
        torch_op=torch.var_mean,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


@pytest.mark.index_select
def test_perf_index_select():
    def index_select_args(dtype, batch, shape):
        inp = torch.randn(shape, dtype=dtype, device="cuda")

        threshold = 0.1
        dim = 0
        index_size = inp.size(dim)
        from math import floor

        index = torch.randint(
            0, index_size, [floor(index_size * threshold)], device="cuda"
        )
        return (inp, dim, index)

    bench = Benchmark(
        op_name="index_select",
        torch_op=torch.index_select,
        arg_func=index_select_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()


@pytest.mark.masked_select
def test_masked_select():
    def masked_select_args(dtype, batch, shape):
        inp = torch.randn(shape, dtype=dtype, device="cuda")
        mask = torch.randn(shape, dtype=dtype, device="cuda") < 0.3
        return (inp, mask)

    bench = Benchmark(
        op_name="masked_select",
        torch_op=torch.masked_select,
        arg_func=masked_select_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=REDUCTION_SHAPES,
    )
    bench.run()
