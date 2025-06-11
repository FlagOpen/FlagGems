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


@pytest.mark.triu
def test_perf_triu():
    bench = Benchmark(
        op_name="triu",
        torch_op=torch.triu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()

