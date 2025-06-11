import torch
import pytest

from .performance_utils import (
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_BATCH,
    SIZES,
    Benchmark,
    binary_args,
    binary_int_args,
    ternary_args,
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

@pytest.mark.abs
def test_perf_abs():
    bench = UnaryPointwiseBenchmark(
        op_name="abs",
        torch_op=torch.abs,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()

@pytest.mark.bitwise_not
def test_perf_bitwisenot():
    bench = UnaryPointwiseBenchmark(
        op_name="bitwisenot_int",
        torch_op=torch.bitwise_not,
        arg_func=unary_int_arg,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.cos
def test_perf_cos():
    bench = UnaryPointwiseBenchmark(
        op_name="cos",
        torch_op=torch.cos,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.dropout
def test_perf_dropout():
    bench = UnaryPointwiseBenchmark(
        op_name="dropout",
        torch_op=torch.nn.Dropout(p=0.5),
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.native_dropout
def test_perf_native_dropout():
    bench = UnaryPointwiseBenchmark(
        op_name="native_dropout",
        torch_op=torch.nn.Dropout(p=0.5),
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.exp
def test_perf_exp():
    bench = UnaryPointwiseBenchmark(
        op_name="exp",
        torch_op=torch.exp,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()




@pytest.mark.gelu
def test_perf_gelu():
    bench = UnaryPointwiseBenchmark(
        op_name="gelu",
        torch_op=torch.nn.functional.gelu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()

@pytest.mark.isinf
def test_perf_isinf():
    bench = UnaryPointwiseBenchmark(
        op_name="isinf",
        torch_op=torch.isinf,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.isnan
def test_perf_isnan():
    bench = UnaryPointwiseBenchmark(
        op_name="isnan",
        torch_op=torch.isnan,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.neg
def test_perf_neg():
    bench = UnaryPointwiseBenchmark(
        op_name="neg",
        torch_op=torch.neg,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.reciprocal
def test_perf_reciprocal():
    bench = UnaryPointwiseBenchmark(
        op_name="reciprocal",
        torch_op=torch.reciprocal,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.relu
def test_perf_relu():
    bench = UnaryPointwiseBenchmark(
        op_name="relu",
        torch_op=torch.nn.functional.relu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.rsqrt
def test_perf_rsqrt():
    bench = UnaryPointwiseBenchmark(
        op_name="rsqrt",
        torch_op=torch.rsqrt,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.sigmoid
def test_perf_sigmoid():
    bench = UnaryPointwiseBenchmark(
        op_name="sigmoid",
        torch_op=torch.sigmoid,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.silu
def test_perf_silu():
    bench = UnaryPointwiseBenchmark(
        op_name="silu",
        torch_op=torch.nn.functional.silu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.sin
def test_perf_sin():
    bench = UnaryPointwiseBenchmark(
        op_name="sin",
        torch_op=torch.sin,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()

    
@pytest.mark.tanh
def test_perf_tanh():
    bench = UnaryPointwiseBenchmark(
        op_name="tanh",
        torch_op=torch.tanh,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()

