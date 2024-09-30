import itertools

import pytest
import torch

from .attri_util import (
    BLAS_BATCH,
    DEFAULT_BATCH,
    DEFAULT_NON_BLAS_BENCH_SHAPES,
    FLOAT_DTYPES,
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


# (N, C, H, W, num_groups)
GROUPNORM_RECCOMMENDED_SHAPES = [
    (16, 16, 8, 8, 16),
    (16, 16, 8, 48, 16),
    (16, 16, 8, 88, 16),
    (16, 16, 8, 128, 16),
    (16, 16, 8, 168, 16),
]


@pytest.mark.groupnorm(recommended_shapes=GROUPNORM_RECCOMMENDED_SHAPES)
def test_perf_groupnorm():
    def group_norm_args(dtype, batch, shape):
        C = shape[1]
        G = shape[-1]
        inp = torch.randn([shape[0], C, shape[2], shape[3]], dtype=dtype, device="cuda")
        weight = torch.randn(
            [
                C,
            ],
            dtype=dtype,
            device="cuda",
        )
        bias = torch.randn(
            [
                C,
            ],
            dtype=dtype,
            device="cuda",
        )
        return inp, G, weight, bias

    bench = Benchmark(
        op_name="groupnorm",
        torch_op=torch.nn.functional.group_norm,
        arg_func=group_norm_args,
        dtypes=FLOAT_DTYPES,
        batch=BLAS_BATCH,
        sizes=GROUPNORM_RECCOMMENDED_SHAPES,
    )
    bench.run()


@pytest.mark.layernorm
def test_perf_layernorm():
    def layer_norm_args(dtype, batch, shape):
        inp = torch.randn(shape, dtype=dtype, device="cuda")
        weight = torch.randn(shape[-1], dtype=dtype, device="cuda")
        bias = torch.randn(shape[-1], dtype=dtype, device="cuda")
        return (
            inp,
            [
                shape[-1],
            ],
            weight,
            bias,
        )

    bench = Benchmark(
        op_name="layernorm",
        torch_op=torch.layer_norm,
        arg_func=layer_norm_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
    )
    bench.run()


@pytest.mark.weight_norm_interface
def test_perf_weightnorm():
    def weight_norm_args(dtype, batch, shape):
        v = torch.randn(shape, dtype=dtype, device="cuda")
        g = torch.randn(shape[0], dtype=dtype, device="cuda")
        return v, g, 0

    bench = Benchmark(
        op_name="weight_norm",
        torch_op=torch._weight_norm_interface,
        arg_func=weight_norm_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
    )
    bench.run()


@pytest.mark.vector_norm
def test_perf_vector_norm():
    bench = Benchmark(
        op_name="vector_norm",
        torch_op=torch.linalg.vector_norm,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
    )
    bench.run()
