import pytest
import torch

import flag_gems

from .attri_util import (
    DEFAULT_BATCH,
    DEFAULT_NON_BLAS_BENCH_SHAPES,
    FLOAT_DTYPES,
    LEGACY_SHAPES,
)
from .performance_utils import Benchmark, binary_args


@pytest.mark.gelu_and_mul
def test_perf_gelu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.gelu(x), y)

    gems_op = flag_gems.gelu_and_mul

    bench = Benchmark(
        op_name="gelu_and_mul",
        torch_op=torch_op,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.silu_and_mul
def test_perf_silu_and_mul():
    def torch_op(x, y):
        return torch.mul(torch.nn.functional.silu(x), y)

    gems_op = flag_gems.silu_and_mul

    bench = Benchmark(
        op_name="silu_and_mul",
        torch_op=torch_op,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.skip_layernorm
def test_perf_skip_layernorm():
    def skip_layernorm_args(dtype, batch, shape):
        inp = torch.randn(shape, dtype=dtype, device="cuda")
        residual = torch.randn(shape, dtype=dtype, device="cuda")
        weight = torch.randn(
            [
                shape[-1],
            ],
            dtype=dtype,
            device="cuda",
        )
        bias = torch.randn(
            [
                shape[-1],
            ],
            dtype=dtype,
            device="cuda",
        )
        return (
            inp,
            residual,
            [
                shape[-1],
            ],
            weight,
            bias,
        )

    def torch_op(inp, residual, layer_shape, weight, bias):
        return torch.layer_norm(inp + residual, layer_shape, weight, bias)

    gems_op = flag_gems.skip_layer_norm

    bench = Benchmark(
        op_name="skip_layernorm",
        torch_op=torch_op,
        arg_func=skip_layernorm_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
    )
    bench.set_gems(gems_op)
    bench.run()


@pytest.mark.skip_rmsnorm
def test_perf_skip_rmsnorm():
    def skip_rmsnorm_args(dtype, batch, shape):
        inp = torch.randn(shape, dtype=dtype, device="cuda")
        residual = torch.randn(shape, dtype=dtype, device="cuda")
        weight = torch.randn(
            [
                shape[-1],
            ],
            dtype=dtype,
            device="cuda",
        )
        return (
            inp,
            residual,
            [
                shape[-1],
            ],
            weight,
            1e-5,
        )

    def torch_op(x, residual, layer_shape, weight, eps):
        x = x + residual
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    gems_op = flag_gems.skip_rms_norm

    bench = Benchmark(
        op_name="skip_rmsnorm",
        torch_op=torch_op,
        arg_func=skip_rmsnorm_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
    )
    bench.set_gems(gems_op)
    bench.run()
