from typing import Generator

import pytest
import torch

from .attri_util import BOOL_DTYPES, FLOAT_DTYPES, INT_DTYPES
from .performance_utils import Benchmark


class UnaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking unary pointwise operations.
    """

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
    ("abs", torch.abs, FLOAT_DTYPES),
    ("erf", torch.erf, FLOAT_DTYPES),
    ("exp", torch.exp, FLOAT_DTYPES),
    ("neg", torch.neg, FLOAT_DTYPES),
    ("reciprocal", torch.reciprocal, FLOAT_DTYPES),
    ("rsqrt", torch.rsqrt, FLOAT_DTYPES),
    ("triu", torch.triu, FLOAT_DTYPES),
    # Activation operations
    ("gelu", torch.nn.functional.gelu, FLOAT_DTYPES),
    ("relu", torch.nn.functional.relu, FLOAT_DTYPES),
    ("sigmoid", torch.sigmoid, FLOAT_DTYPES),
    ("silu", torch.nn.functional.silu, FLOAT_DTYPES),
    # Trigonometric operations
    ("cos", torch.cos, FLOAT_DTYPES),
    ("sin", torch.sin, FLOAT_DTYPES),
    ("tanh", torch.tanh, FLOAT_DTYPES),
    # Bitwise operations
    ("bitwise_not", torch.bitwise_not, INT_DTYPES),
    # Numerical Checks
    ("isinf", torch.isinf, FLOAT_DTYPES),
    ("isnan", torch.isnan, FLOAT_DTYPES),
    ("isfinite", torch.isfinite, FLOAT_DTYPES + INT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(name, op, dtype, marks=getattr(pytest.mark, name, None))
        for name, op, dtype in forward_operations
    ],
)
def test_general_unary_pointwise_perf(op_name, torch_op, dtypes):
    bench = UnaryPointwiseBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()


backward_operations = [
    ("gelu", torch.nn.functional.gelu, FLOAT_DTYPES),
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
def test_general_unary_pointwise_backward_perf(op_name, torch_op, dtypes):
    bench = UnaryPointwiseBenchmark(
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
        is_backward=True,
    )
    bench.run()
