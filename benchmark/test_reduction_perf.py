import torch

from .performance_utils import (
    BLAS_BATCH,
    FLOAT_DTYPES,
    REDUCTION_BATCH,
    SIZES,
    Benchmark,
    unary_arg,
)


def test_perf_all():
    bench = Benchmark(
        op_name="all",
        torch_op=torch.all,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_amax():
    bench = Benchmark(
        op_name="amax",
        torch_op=torch.amax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_any():
    bench = Benchmark(
        op_name="any",
        torch_op=torch.any,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_argmax():
    bench = Benchmark(
        op_name="argmax",
        torch_op=torch.argmax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_cross_entropy_loss():
    def cross_entropy_loss_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device="cuda")
        target = torch.randint(
            0,
            size,
            [
                batch,
            ],
            device="cuda",
        )
        return inp, target

    bench = Benchmark(
        op_name="cross_entropy_loss",
        torch_op=torch.nn.CrossEntropyLoss(),
        arg_func=cross_entropy_loss_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_cumsum():
    def cumsum_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device="cuda")
        return inp, 1

    bench = Benchmark(
        op_name="cumsum",
        torch_op=torch.cumsum,
        arg_func=cumsum_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_groupnorm():
    def group_norm_args(dtype, batch, size):
        C = 16
        G = 16
        inp = torch.randn([batch, C, size], dtype=dtype, device="cuda")
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
        sizes=SIZES,
    )
    bench.run()


def test_perf_layernorm():
    def layer_norm_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device="cuda")
        weight = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device="cuda",
        )
        bias = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device="cuda",
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
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_log_softmax():
    bench = Benchmark(
        op_name="log_softmax",
        torch_op=torch.nn.functional.log_softmax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_max():
    bench = Benchmark(
        op_name="max",
        torch_op=torch.max,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_mean():
    bench = Benchmark(
        op_name="mean",
        torch_op=torch.mean,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_min():
    bench = Benchmark(
        op_name="min",
        torch_op=torch.min,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_prod():
    bench = Benchmark(
        op_name="prod",
        torch_op=torch.prod,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_softmax():
    bench = Benchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_softmax_backward():
    bench = Benchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
        is_backward=True,
    )
    bench.run()


def test_perf_sum():
    bench = Benchmark(
        op_name="sum",
        torch_op=torch.sum,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_var_mean():
    bench = Benchmark(
        op_name="var_mean",
        torch_op=torch.var_mean,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_vector_norm():
    bench = Benchmark(
        op_name="vector_norm",
        torch_op=torch.linalg.vector_norm,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()
