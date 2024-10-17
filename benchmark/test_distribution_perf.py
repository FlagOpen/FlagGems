import pytest
import torch

from .attri_util import DEFAULT_BINARY_POINTWISE_SHAPES, FLOAT_DTYPES
from .performance_utils import GenericBenchmark, unary_input_fn


def normal_input_fn(shape, cur_dtype, device):
    loc = torch.full(shape, fill_value=3.0, dtype=cur_dtype, device=device)
    scale = torch.full(shape, fill_value=10.0, dtype=cur_dtype, device=device)
    yield loc, scale


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "normal",
            torch.distributions.normal.Normal,
            normal_input_fn,
            marks=pytest.mark.normal(
                recommended_shapes=DEFAULT_BINARY_POINTWISE_SHAPES,
                shape_desc="(B), M, N",
            ),
        ),
        pytest.param(
            "uniform_",
            torch.Tensor.uniform_,
            unary_input_fn,
            marks=pytest.mark.uniform_(
                recommended_shapes=DEFAULT_BINARY_POINTWISE_SHAPES,
                shape_desc="(B), M, N",
            ),
        ),
        pytest.param(
            "exponential_",
            torch.Tensor.exponential_,
            unary_input_fn,
            marks=pytest.mark.exponential_(
                recommended_shapes=DEFAULT_BINARY_POINTWISE_SHAPES,
                shape_desc="(B), M, N",
            ),
        ),
    ],
)
def test_distribution_benchmark(op_name, torch_op, input_fn):
    bench = GenericBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.DEFAULT_SHAPES = DEFAULT_BINARY_POINTWISE_SHAPES
    bench.run()
