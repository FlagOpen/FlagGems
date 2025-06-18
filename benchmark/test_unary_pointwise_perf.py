import torch
import pytest
from benchmark.op_configs import op_configs

from .performance_utils import (
    Benchmark,
    unary_arg,
    unary_int_arg,
    device,
    DEFAULT_METRICS
)

class UnaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking unary pointwise operations.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "abs"])
@pytest.mark.abs
def test_perf_abs(config):
    bench = UnaryPointwiseBenchmark(
        op_name="abs",
        torch_op=torch.abs,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "bitwise_not"])
@pytest.mark.bitwise_not
def test_perf_bitwisenot(config):
    bench = UnaryPointwiseBenchmark(
        op_name="bitwisenot_int",
        torch_op=torch.bitwise_not,
        arg_func=unary_int_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "cos"])
@pytest.mark.cos
def test_perf_cos(config):
    bench = UnaryPointwiseBenchmark(
        op_name="cos",
        torch_op=torch.cos,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "dropout"])
@pytest.mark.dropout
def test_perf_dropout(config):
    bench = UnaryPointwiseBenchmark(
        op_name="dropout",
        torch_op=torch.nn.Dropout(p=0.5),
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "native_dropout"])
@pytest.mark.native_dropout
def test_perf_native_dropout(config):
    bench = UnaryPointwiseBenchmark(
        op_name="native_dropout",
        torch_op=torch.nn.Dropout(p=0.5),
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "exp"])
@pytest.mark.exp
def test_perf_exp(config):
    bench = UnaryPointwiseBenchmark(
        op_name="exp",
        torch_op=torch.exp,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "gelu"])
@pytest.mark.gelu
def test_perf_gelu(config):
    bench = UnaryPointwiseBenchmark(
        op_name="gelu",
        torch_op=torch.nn.functional.gelu,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "isinf"])
@pytest.mark.isinf
def test_perf_isinf(config):
    bench = UnaryPointwiseBenchmark(
        op_name="isinf",
        torch_op=torch.isinf,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "isnan"])
@pytest.mark.isnan
def test_perf_isnan(config):
    bench = UnaryPointwiseBenchmark(
        op_name="isnan",
        torch_op=torch.isnan,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "isnan"])
@pytest.mark.neg
def test_perf_neg(config):
    bench = UnaryPointwiseBenchmark(
        op_name="neg",
        torch_op=torch.neg,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "reciprocal"])
@pytest.mark.reciprocal
def test_perf_reciprocal(config):
    bench = UnaryPointwiseBenchmark(
        op_name="reciprocal",
        torch_op=torch.reciprocal,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "relu"])
@pytest.mark.relu
def test_perf_relu(config):
    bench = UnaryPointwiseBenchmark(
        op_name="relu",
        torch_op=torch.nn.functional.relu,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "rsqrt"])
@pytest.mark.rsqrt
def test_perf_rsqrt(config):
    bench = UnaryPointwiseBenchmark(
        op_name="rsqrt",
        torch_op=torch.rsqrt,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "sigmoid"])
@pytest.mark.sigmoid
def test_perf_sigmoid(config):
    bench = UnaryPointwiseBenchmark(
        op_name="sigmoid",
        torch_op=torch.sigmoid,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "silu"])
@pytest.mark.silu
def test_perf_silu(config):
    bench = UnaryPointwiseBenchmark(
        op_name="silu",
        torch_op=torch.nn.functional.silu,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "sin"])
@pytest.mark.sin
def test_perf_sin(config):
    bench = UnaryPointwiseBenchmark(
        op_name="sin",
        torch_op=torch.sin,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "tanh"])
@pytest.mark.tanh
def test_perf_tanh(config):
    bench = UnaryPointwiseBenchmark(
        op_name="tanh",
        torch_op=torch.tanh,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

