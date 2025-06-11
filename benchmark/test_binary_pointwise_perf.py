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


@pytest.mark.add
def test_perf_add():
    bench = BinaryPointwiseBenchmark(
        op_name="add",
        torch_op=torch.add,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.bitwise_and
def test_perf_bitwiseand():
    bench = BinaryPointwiseBenchmark(
        op_name="bitwiseand_int",
        torch_op=torch.bitwise_and,
        arg_func=binary_int_args,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()



@pytest.mark.bitwise_or
def test_perf_bitwiseor():
    bench = BinaryPointwiseBenchmark(
        op_name="bitwiseor_int",
        torch_op=torch.bitwise_or,
        arg_func=binary_int_args,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()



@pytest.mark.div
def test_perf_div():
    bench = BinaryPointwiseBenchmark(
        op_name="div",
        torch_op=torch.div,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()



@pytest.mark.eq
def test_perf_eq():
    bench = BinaryPointwiseBenchmark(
        op_name="eq",
        torch_op=torch.eq,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.ge
def test_perf_ge():
    bench = BinaryPointwiseBenchmark(
        op_name="ge",
        torch_op=torch.ge,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()




@pytest.mark.gt
def test_perf_gt():
    bench = BinaryPointwiseBenchmark(
        op_name="gt",
        torch_op=torch.gt,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()



@pytest.mark.le
def test_perf_le():
    bench = BinaryPointwiseBenchmark(
        op_name="le",
        torch_op=torch.le,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.lt
def test_perf_lt():
    bench = BinaryPointwiseBenchmark(
        op_name="lt",
        torch_op=torch.lt,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.mul
def test_perf_mul():
    bench = BinaryPointwiseBenchmark(
        op_name="mul",
        torch_op=torch.mul,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.ne
def test_perf_ne():
    bench = BinaryPointwiseBenchmark(
        op_name="ne",
        torch_op=torch.ne,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.pow
def test_perf_pow():
    bench = BinaryPointwiseBenchmark(
        op_name="pow",
        torch_op=torch.pow,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.sub
def test_perf_sub():
    bench = BinaryPointwiseBenchmark(
        op_name="sub",
        torch_op=torch.sub,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.rsub
def test_perf_rsub():
    bench = BinaryPointwiseBenchmark(
        op_name="rsub",
        torch_op=torch.rsub,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()
    