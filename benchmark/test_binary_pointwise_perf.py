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
from .performance_utils import Benchmark


class BinaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking binary pointwise operations.
    """

    def set_shapes(self):
        self.shapes = DEFAULT_NON_BLAS_BENCH_SHAPES[:]
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            # TODO: Currently, we are not considering the following scenarios:
            # - Tensor and scale operations
            # - Scale and scale operations
            # - Scenarios where alpha and beta are not set to their default values
            # It is under discussion whether these scenarios should be included in the comprehensive level of testing.
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
            inp1 = self._generate_inputs(shape, cur_dtype)
            inp2 = self._generate_inputs(shape, cur_dtype)
            yield inp1, inp2


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(name, op, dtype, marks=getattr(pytest.mark, name, None))
        for name, op, dtype in [
            # Arithmetic operations
            ("add", torch.add, FLOAT_DTYPES),
            ("div", torch.div, FLOAT_DTYPES),
            ("mul", torch.mul, FLOAT_DTYPES),
            ("pow", torch.pow, FLOAT_DTYPES),
            ("sub", torch.sub, FLOAT_DTYPES),
            ("floor_divide", torch.floor_divide, INT_DTYPES),
            ("remainder", torch.remainder, INT_DTYPES),
            # Comparison operations
            ("eq", torch.eq, FLOAT_DTYPES),
            ("ge", torch.ge, FLOAT_DTYPES),
            ("gt", torch.gt, FLOAT_DTYPES),
            ("le", torch.le, FLOAT_DTYPES),
            ("lt", torch.lt, FLOAT_DTYPES),
            ("ne", torch.ne, FLOAT_DTYPES),
            # Minimum and maximum operations
            ("maximum", torch.maximum, FLOAT_DTYPES),
            ("minimum", torch.minimum, FLOAT_DTYPES),
            # Bitwise operations
            ("bitwise_and", torch.bitwise_and, INT_DTYPES + BOOL_DTYPES),
            ("bitwise_or", torch.bitwise_or, INT_DTYPES + BOOL_DTYPES),
            # Numerical Checks
            ("isclose", torch.isclose, FLOAT_DTYPES + INT_DTYPES),
            ("allclose", torch.allclose, FLOAT_DTYPES + INT_DTYPES),
        ]
    ],
)
def test_general_binary_pointwise_perf(op_name, torch_op, dtypes):
    bench = BinaryPointwiseBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()
