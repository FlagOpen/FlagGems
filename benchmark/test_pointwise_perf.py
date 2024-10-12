import itertools
from typing import Generator

import pytest
import torch

from .attri_util import DEFAULT_NON_BLAS_BENCH_SHAPES, FLOAT_DTYPES, INT_DTYPES
from .conftest import BenchLevel, Config
from .performance_utils import Benchmark, GenericBenchmark, generate_tensor_input


def flip_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    yield inp, {"dims": [0, 1]}


def masked_fill_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    mask = generate_tensor_input(shape, cur_dtype, device) < 0.3
    value = 1024
    yield inp, mask, value


def tile_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    yield inp, {
        "dims": [2, 4]
    }  # TODO: Fails when encountering certain corner shape cases.


def repeat_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = [2, 4]
    yield inp1, inp2,  # TODO: Fails when encountering certain corner shape cases.


def where_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = generate_tensor_input(shape, cur_dtype, device)
    condition = inp1 > 0
    yield condition, inp1, inp2


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "flip",
            torch.flip,
            flip_input_fn,
            FLOAT_DTYPES + INT_DTYPES,
            marks=pytest.mark.flip,
        ),
        pytest.param(
            "masked_fill",
            torch.masked_fill,
            masked_fill_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.masked_fill,
        ),
        pytest.param(
            "tile", torch.tile, tile_input_fn, FLOAT_DTYPES, marks=pytest.mark.tile
        ),
        pytest.param(
            "repeat",
            torch.Tensor.repeat,
            repeat_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.repeat,
        ),
        pytest.param(
            "where", torch.where, where_input_fn, FLOAT_DTYPES, marks=pytest.mark.where
        ),
    ],
)
def test_generic_pointwise_benchmark(op_name, torch_op, input_fn, dtypes):
    bench = GenericBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=dtypes
    )
    bench.run()


class ClampBenchmark(Benchmark):
    """
    benchmark for clamp
    """

    def set_shapes(self):
        # self.shapes is a list of tuples, where each tuple consists of three elements:
        # 1. The first element represents the shape of the input tensor.
        # 2. The second element represents the information for "min",
        #    which can be a tensor shape, a scalar value, or None.
        # 3. The third element represents the information for "max",
        #   which can be a tensor shape, a scalar value, or None.
        self.shapes = [(shape, shape, shape) for shape in DEFAULT_NON_BLAS_BENCH_SHAPES]
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            more_shapes = [(1024, 1024), (3, 240, 5)]
            scalars = [0.001, -111.999]
            self.shapes.extend([(shape, None, shape) for shape in more_shapes])
            self.shapes.extend(
                [
                    (shape, scalar, scalar)
                    for shape, scalar in itertools.product(more_shapes, scalars)
                ]
            )
            self.shapes.extend(
                [
                    (shape, scalar, None)
                    for shape, scalar in itertools.product(more_shapes, scalars)
                ]
            )

    def init_inp(self, shape, cur_dtype):
        if isinstance(shape, (list, tuple)):
            return torch.randn(shape, dtype=cur_dtype, device=self.device)
        elif shape is None:
            return None
        elif isinstance(shape, float):
            return shape  # scalar value

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape1, shape2, shape3 in self.shapes:
            inp1 = self.init_inp(shape1, cur_dtype)
            inp2 = self.init_inp(shape2, cur_dtype)
            inp3 = self.init_inp(shape3, cur_dtype)
            yield inp1, inp2, inp3


@pytest.mark.clamp
def test_perf_clamp():
    bench = ClampBenchmark(
        op_name="clamp",
        torch_op=torch.clamp,
    )
    bench.run()
