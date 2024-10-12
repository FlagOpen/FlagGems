from typing import Generator

import pytest
import torch

from .attri_util import (
    BOOL_DTYPES,
    DEFAULT_BINARY_POINTWISE_SHAPES,
    DEFAULT_NON_BLAS_BENCH_SHAPES,
    FLOAT_DTYPES,
    INT_DTYPES,
)
from .conftest import BenchLevel, Config
from .performance_utils import Benchmark, generate_tensor_input

special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
shapes_3d = [(shape[0], *shape) for shape in DEFAULT_NON_BLAS_BENCH_SHAPES]
sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
COMPREHENSIVE_SHAPES = set(
    DEFAULT_BINARY_POINTWISE_SHAPES
    + DEFAULT_NON_BLAS_BENCH_SHAPES
    + special_shapes_2d
    + shapes_3d
    + sp_shapes_3d
)


class BinaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking binary pointwise operations.
    """

    def set_shapes(self):
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            self.shapes = COMPREHENSIVE_SHAPES
        else:
            self.shapes = DEFAULT_BINARY_POINTWISE_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp1 = generate_tensor_input(shape, cur_dtype, self.device)
            inp2 = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp1, inp2


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name, None)(
                recommended_shapes=DEFAULT_BINARY_POINTWISE_SHAPES,
                shape_desc="(B), M, N",
            ),
        )
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
