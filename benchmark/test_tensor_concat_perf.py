from typing import Generator

import pytest
import torch

from .attri_util import FLOAT_DTYPES, INT_DTYPES, BenchLevel
from .performance_utils import Benchmark, Config, generate_tensor_input

CONCATENATION_RECOMMENDED_SHAPES = [
    (512, 64),
    (512, 384),
    (512, 704),
    (512, 1024),
    (512, 1344),
]


class ConcatBenchmark(Benchmark):
    """
    benchmark for concat and stack
    """

    DEFAULT_SHAPES = CONCATENATION_RECOMMENDED_SHAPES

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)

    def set_shapes(self):
        self.shapes = self.DEFAULT_SHAPES[:]
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            # TODO: ADD MORE shapes for concat
            self.shapes.extend([])


def stack_cat_input_fn(shape, dtype, device):
    inp1 = generate_tensor_input(shape, dtype, device)
    inp2 = generate_tensor_input(shape, dtype, device)
    inp3 = generate_tensor_input(shape, dtype, device)
    yield [inp1, inp2, inp3], {"dim": 0},
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield [inp1, inp2, inp3], {"dim": -1},


def hstack_vstack_input_fn(shape, dtype, device):
    inp1 = generate_tensor_input(shape, dtype, device)
    inp2 = generate_tensor_input(shape, dtype, device)
    inp3 = generate_tensor_input(shape, dtype, device)
    yield [inp1, inp2, inp3],


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtype",
    [
        pytest.param(
            "cat",
            torch.cat,
            stack_cat_input_fn,
            FLOAT_DTYPES + INT_DTYPES,
            marks=pytest.mark.cat(recommended_shapes=CONCATENATION_RECOMMENDED_SHAPES, shape_desc="((B), M, N) * 3"
            ),
        ),
        pytest.param(
            "stack",
            torch.stack,
            stack_cat_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.stack(
                recommended_shapes=CONCATENATION_RECOMMENDED_SHAPES, shape_desc="((B), M, N) * 3"
            ),
        ),
        pytest.param(
            "hstack",
            torch.hstack,
            hstack_vstack_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.hstack(
                recommended_shapes=CONCATENATION_RECOMMENDED_SHAPES, shape_desc="((B), M, N) * 3"
            ),
        ),
        pytest.param(
            "vstack",
            torch.vstack,
            hstack_vstack_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.vstack(
                recommended_shapes=CONCATENATION_RECOMMENDED_SHAPES, shape_desc="((B), M, N) * 3"
            ),
        ),
    ],
)
def test_concat_benchmark(op_name, torch_op, input_fn, dtype):
    bench = ConcatBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=dtype
    )
    bench.run()
