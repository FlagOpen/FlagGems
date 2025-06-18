import torch
import pytest
from benchmark.op_configs import op_configs
from .performance_utils import (
    Benchmark,
    unary_arg,
    get_shape,
    device
)

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "all"])
@pytest.mark.all
def test_perf_all(config):
    bench = Benchmark(
        op_name="all",
        torch_op=torch.all,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        need_dim=True,
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "amax"])
@pytest.mark.amax
def test_perf_amax(config):
    bench = Benchmark(
        op_name="amax",
        torch_op=torch.amax,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        need_dim=True,
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "any"])
def test_perf_any(config):
    bench = Benchmark(
        op_name="any",
        torch_op=torch.any,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "argmax"])
@pytest.mark.argmax
def test_perf_argmax(config):
    def argmax_kwargs(dtype,batch,size):
        return {"keep_dim": False}
    bench = Benchmark(
        op_name="argmax",
        torch_op=torch.argmax,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        need_dim=True,
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "CrossEntropyLoss"])
@pytest.mark.CrossEntropyLoss
def test_perf_CrossEntropyLoss(config):
    def cross_entropy_loss_args(dtype, batch, size):
        shape = get_shape(batch, size)
        inp = torch.randn(shape, dtype=dtype, device=device)
        target = torch.randint(0, shape[1], [shape[0],], device=device,)
        return inp, target

    bench = Benchmark(
        op_name="CrossEntropyLoss",
        torch_op=torch.nn.CrossEntropyLoss(),
        arg_func=cross_entropy_loss_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "cumsum"])
@pytest.mark.cumsum
def test_perf_cumsum(config):
    def cumsum_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device=device)
        return inp, 1

    bench = Benchmark(
        op_name="cumsum",
        torch_op=torch.cumsum,
        arg_func=cumsum_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "group_norm"])
@pytest.mark.group_norm
def test_perf_group_norm(config):
    def group_norm_args(dtype, batch, size):
        shape = get_shape(batch, size)
        G = 8
        if shape[1] == 6:
            G=3
        inp = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(
            [shape[1]],
            dtype=dtype,
            device=device,
        )
        bias = torch.randn(
            [shape[1]],
            dtype=dtype,
            device=device,
        )
        return inp, G, weight, bias

    bench = Benchmark(
        op_name="groupnorm",
        torch_op=torch.nn.functional.group_norm,
        arg_func=group_norm_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "layernorm"])
@pytest.mark.layernorm
def test_perf_layernorm(config):
    def layer_norm_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device=device)
        weight = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device=device,
        )
        bias = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device=device,
        )
        return (
            inp,
            [
                size,
            ],
            weight,
            bias,
        )

    bench = Benchmark(
        op_name="layernorm",
        torch_op=torch.layer_norm,
        arg_func=layer_norm_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "log_softmax"])
@pytest.mark.log_softmax
def test_perf_log_softmax(config):
    bench = Benchmark(
        op_name="log_softmax",
        torch_op=torch.nn.functional.log_softmax,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "max"])
@pytest.mark.max
def test_perf_max(config):
    bench = Benchmark(
        op_name="max",
        torch_op=torch.max,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        need_dim=True,
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "mean"])
@pytest.mark.mean
def test_perf_mean(config):
    bench = Benchmark(
        op_name="mean",
        torch_op=torch.mean,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        need_dim=True,
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "min"])
@pytest.mark.min
def test_perf_min(config):
    bench = Benchmark(
        op_name="min",
        torch_op=torch.min,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        need_dim=True,
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "prod"])
@pytest.mark.prod
def test_perf_prod(config):
    bench = Benchmark(
        op_name="prod",
        torch_op=torch.prod,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        need_dim=True,
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "softmax"])
@pytest.mark.softmax
def test_perf_softmax(config):
    bench = Benchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        need_dim=True,
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "softmax_backward"])
@pytest.mark.softmax_backward
def test_perf_softmax_backward(config):
    bench = Benchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        is_backward=True,
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "sum"])
@pytest.mark.sum
def test_perf_sum(config):
    bench = Benchmark(
        op_name="sum",
        torch_op=torch.sum,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
        need_dim=True,
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "var_mean"])
@pytest.mark.var_mean
def test_perf_var_mean(config):
    bench = Benchmark(
        op_name="var_mean",
        torch_op=torch.var_mean,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "vector_norm"])
@pytest.mark.vector_norm
def test_perf_vector_norm(config):
    bench = Benchmark(
        op_name="vector_norm",
        torch_op=torch.linalg.vector_norm,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()
