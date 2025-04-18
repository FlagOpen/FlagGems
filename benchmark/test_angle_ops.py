from typing import Generator

import pytest
import torch
import flag_gems

from .attri_util import BOOL_DTYPES, DEFAULT_METRICS, FLOAT_DTYPES, INT_DTYPES,COMPLEX_DTYPES
from .performance_utils import Benchmark, generate_tensor_input, vendor_name


class UnaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking unary pointwise operations.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp,

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()
forward_operations = [
    ("angle", torch.angle, [torch.complex32]+[torch.complex64]+[torch.float32]),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name, None),
        )
        for name, op, dtype in forward_operations
    ],
)
def test_general_unary_pointwise_perf(op_name, torch_op, dtypes):
    if vendor_name == "kunlunxin" and op_name == "elu":
        pytest.skip("RUNTIME TODOFIX")
    bench = UnaryPointwiseBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()

