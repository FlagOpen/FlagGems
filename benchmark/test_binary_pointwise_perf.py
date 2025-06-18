import torch
import pytest
from benchmark.op_configs import op_configs

from .performance_utils import (
    Benchmark,
    binary_args,
    binary_int_args,
    device,
    DEFAULT_METRICS
)

class BinaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking binary pointwise operations.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def get_tflops(self, op, *args, **kwargs):
        shape1 = list(args[0].shape)
        shape2 = list(args[0].shape)
        return torch.tensor(shape1).prod().item() + torch.tensor(shape2).prod().item()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "add"])
@pytest.mark.add
def test_perf_add(config):
    bench = BinaryPointwiseBenchmark(
        op_name="add",
        torch_op=torch.add,
        arg_func=binary_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "bitwise_and"])
@pytest.mark.bitwise_and
def test_perf_bitwiseand(config):
    bench = BinaryPointwiseBenchmark(
        op_name="bitwise_and",
        torch_op=torch.bitwise_and,
        arg_func=binary_int_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()


@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "bitwise_or"])
@pytest.mark.bitwise_or
def test_perf_bitwiseor(config):
    bench = BinaryPointwiseBenchmark(
        op_name="bitwiseor_int",
        torch_op=torch.bitwise_or,
        arg_func=binary_int_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()


@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "div"])
@pytest.mark.div
def test_perf_div(config):
    bench = BinaryPointwiseBenchmark(
        op_name="div",
        torch_op=torch.div,
        arg_func=binary_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()


@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "eq"])
@pytest.mark.eq
def test_perf_eq(config):
    bench = BinaryPointwiseBenchmark(
        op_name="eq",
        torch_op=torch.eq,
        arg_func=binary_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "ge"])
@pytest.mark.ge
def test_perf_ge(config):
    bench = BinaryPointwiseBenchmark(
        op_name="ge",
        torch_op=torch.ge,
        arg_func=binary_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()



@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "gt"])
@pytest.mark.gt
def test_perf_gt(config):
    bench = BinaryPointwiseBenchmark(
        op_name="gt",
        torch_op=torch.gt,
        arg_func=binary_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()


@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "le"])
@pytest.mark.le
def test_perf_le(config):
    bench = BinaryPointwiseBenchmark(
        op_name="le",
        torch_op=torch.le,
        arg_func=binary_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "lt"])
@pytest.mark.lt
def test_perf_lt(config):
    bench = BinaryPointwiseBenchmark(
        op_name="lt",
        torch_op=torch.lt,
        arg_func=binary_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "mul"])
@pytest.mark.mul
def test_perf_mul(config):
    bench = BinaryPointwiseBenchmark(
        op_name="mul",
        torch_op=torch.mul,
        arg_func=binary_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "ne"])
@pytest.mark.ne
def test_perf_ne(config):
    bench = BinaryPointwiseBenchmark(
        op_name="ne",
        torch_op=torch.ne,
        arg_func=binary_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "pow"])
@pytest.mark.pow
def test_perf_pow(config):
    bench = BinaryPointwiseBenchmark(
        op_name="pow",
        torch_op=torch.pow,
        arg_func=binary_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "sub"])
@pytest.mark.sub
def test_perf_sub(config):
    bench = BinaryPointwiseBenchmark(
        op_name="sub",
        torch_op=torch.sub,
        arg_func=binary_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "rsub"])
@pytest.mark.rsub
def test_perf_rsub(config):
    bench = BinaryPointwiseBenchmark(
        op_name="rsub",
        torch_op=torch.rsub,
        arg_func=binary_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()
    