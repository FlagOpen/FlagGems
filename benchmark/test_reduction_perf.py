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


class UnaryReductionBenchmark(Benchmark):
    """
    Base class for benchmarking reduction operations.
    """

    def set_shapes(self):
        self.shapes = DEFAULT_NON_BLAS_BENCH_SHAPES[:]
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            MORE_SHAPES = [(320, 15), (128, 64, 60)]
            MORE_BATCHS = [4, 20, 32]
            combinations = [
                (batch, *shape)
                for batch, shape in itertools.product(MORE_BATCHS, MORE_SHAPES)
            ]
            self.shapes.extend(combinations)

    def _generate_inputs(self, shape, cur_dtype):
        if cur_dtype in FLOAT_DTYPES:
            return torch.randn(shape, dtype=cur_dtype, device=self.device)
        elif cur_dtype in INT_DTYPES:
            return torch.randint(
                torch.iinfo(cur_dtype).min,
                torch.iinfo(cur_dtype).max,
                shape,
                dtype=cur_dtype,
                device=self.device,
            )
        elif cur_dtype in BOOL_DTYPES:
            return torch.randint(0, 2, size=shape, dtype=cur_dtype, device=self.device)

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = self._generate_inputs(shape, cur_dtype)
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
