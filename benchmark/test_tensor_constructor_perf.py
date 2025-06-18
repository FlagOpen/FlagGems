import torch
import pytest
from benchmark.op_configs import op_configs

from .performance_utils import (
    Benchmark,
    unary_arg,
    get_shape,
    device
)

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "ones"])
@pytest.mark.ones
def test_perf_ones(config):
    def ones_kwargs(dtype, batch, size):
        return {"size": (batch, size), "dtype": dtype, "device": "cuda"}

    bench = Benchmark(
        op_name="ones",
        torch_op=torch.ones,
        arg_func=None,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        kwargs_func=ones_kwargs,
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "zeros"])
@pytest.mark.zeros
def test_perf_zeros(config):
    def zeros_kwargs(dtype, batch, size):
        return {"size": (batch, size), "dtype": dtype, "device": "cuda"}

    bench = Benchmark(
        op_name="zeros",
        torch_op=torch.zeros,
        arg_func=None,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        kwargs_func=zeros_kwargs,
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "full"])
@pytest.mark.full
def test_perf_full(config):
    def full_kwargs(dtype, batch, size):
        return {
            "size": (batch, size),
            "fill_value": 3.1415926,
            "dtype": dtype,
            "device": "cuda",
        }

    bench = Benchmark(
        op_name="full",
        torch_op=torch.full,
        arg_func=None,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        kwargs_func=full_kwargs,
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "ones_like"])
@pytest.mark.ones_like
def test_perf_ones_like(config):
    bench = Benchmark(
        op_name="ones_like",
        torch_op=torch.ones_like,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "zeros_like"])
@pytest.mark.zeros_like
def test_perf_zeros_like(config):
    bench = Benchmark(
        op_name="zeros_like",
        torch_op=torch.zeros_like,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "zeros"])
@pytest.mark.zeros
def test_perf_full_like(config):
    def full_kwargs(dtype, batch, size):
        return {
            "input": torch.randn([batch, size], dtype=dtype, device="cuda"),
            "fill_value": 3.1415926,
        }

    bench = Benchmark(
        op_name="full_like",
        torch_op=torch.full_like,
        arg_func=None,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        kwargs_func=full_kwargs,
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "fill"])
@pytest.mark.fill
def test_perf_fill(config):
    def fill_args(dtype, batch, size):
        shape=get_shape(batch, size)
        inp1 = torch.rand(shape, dtype=dtype, device=device)
        inp2 = torch.tensor(3.1415926, dtype=dtype, device=device)
        return inp1,inp2


    bench = Benchmark(
        op_name="fill",
        torch_op=torch.fill,
        arg_func=fill_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        kwargs_func=None,
    )
    bench.run()