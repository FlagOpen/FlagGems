import itertools
from typing import Generator

import pytest
import torch

from .attri_util import (
    BOOL_DTYPES,
    DEFAULT_NON_BLAS_BENCH_SHAPES,
    FLOAT_DTYPES,
    INT_DTYPES,
)
from .conftest import BenchLevel, Config
from .performance_utils import Benchmark, GenericBenchmark, generate_tensor_input

sp_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
shapes_3d = [(shape[0], *shape) for shape in DEFAULT_NON_BLAS_BENCH_SHAPES]
shapes_1d = [(1024 * 1024 * 1024)]
sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
COMPREHENSIVE_SHAPES = list(dict.fromkeys(
    DEFAULT_NON_BLAS_BENCH_SHAPES
    + sp_shapes_2d
    + shapes_3d
    + sp_shapes_3d
    + shapes_1d
))

class UnaryReductionBenchmark(Benchmark):
    """
    Base class for benchmarking reduction operations.
    """
    def set_shapes(self):
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            self.shapes = COMPREHENSIVE_SHAPES
        else:
            self.shapes = DEFAULT_NON_BLAS_BENCH_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp,


forward_operations = [
    # TODO: Set the `keepdim` and `dim` parameters when the benchmark level is set to comprehensive.
    ("all", torch.all, FLOAT_DTYPES),
    ("amax", torch.amax, FLOAT_DTYPES),
    ("any", torch.any, FLOAT_DTYPES),
    ("argmax", torch.argmax, FLOAT_DTYPES),
    ("log_softmax", torch.nn.functional.log_softmax, FLOAT_DTYPES),
    ("max", torch.max, FLOAT_DTYPES),
    ("mean", torch.mean, FLOAT_DTYPES),
    ("min", torch.min, FLOAT_DTYPES),
    ("nonzero", torch.nonzero, FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES),
    ("prod", torch.prod, FLOAT_DTYPES),
    ("softmax", torch.nn.functional.softmax, FLOAT_DTYPES),
    ("sum", torch.sum, FLOAT_DTYPES),
    ("var_mean", torch.var_mean, FLOAT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(name, op, dtype, marks=getattr(pytest.mark, name, None))
        for name, op, dtype in forward_operations
    ],
)
def test_general_reduction_perf(op_name, torch_op, dtypes):
    bench = UnaryReductionBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()


backward_operations = [
    ("softmax", torch.nn.functional.softmax, FLOAT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name, op, dtype, marks=getattr(pytest.mark, name + "_backward", None)
        )
        for name, op, dtype in backward_operations
    ],
)
def test_general_reduction_backward_perf(op_name, torch_op, dtypes):
    bench = UnaryReductionBenchmark(
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
        is_backward=True,
    )
    bench.run()


def cross_entropy_loss_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    target = torch.randint(0, shape[-1], (shape[0],), device=device)
    yield inp, target


def cumsum_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    yield inp, 1


def index_select_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    threshold = 0.1
    dim = 0
    index_size = inp.size(dim)
    from math import floor

    index = torch.randint(0, index_size, [floor(index_size * threshold)], device=device)
    yield inp, dim, index


def masked_select_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    mask = generate_tensor_input(shape, cur_dtype, device) < 0.3
    yield inp, mask


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "CrossEntropyLoss",
            torch.nn.CrossEntropyLoss(),
            cross_entropy_loss_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.CrossEntropyLoss,
        ),
        pytest.param(
            "cumsum",
            torch.cumsum,
            cumsum_input_fn,
            FLOAT_DTYPES + INT_DTYPES,
            marks=pytest.mark.cumsum,
        ),
        pytest.param(
            "index_select",
            torch.index_select,
            index_select_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.index_select,
        ),
        pytest.param(
            "masked_select",
            torch.masked_select,
            masked_select_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.masked_select,
        ),
    ],
)
def test_generic_reduction_benchmark(op_name, torch_op, input_fn, dtypes):
    bench = GenericBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=dtypes
    )
    bench.run()
