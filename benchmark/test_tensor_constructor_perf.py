import pytest
import torch

from .performance_utils import GenericBenchmark, unary_input_fn


def generic_constructor_input_fn(shape, dtype, device):
    yield {"size": shape, "dtype": dtype, "device": device},


def full_input_fn(shape, dtype, device):
    yield {"size": shape, "fill_value": 3.1415926, "dtype": dtype, "device": device},


def full_like_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield {"input": inp, "fill_value": 3.1415926},


def fill_input_fn(shape, dtype, device):
    input = torch.empty(shape, dtype=dtype, device=device)
    yield input, 3.14159,

# Define operations and their corresponding input functions
tensor_constructor_operations = [
    # generic tensor constructor
    ("rand", torch.rand, generic_constructor_input_fn),
    ("randn", torch.randn, generic_constructor_input_fn),
    ("ones", torch.ones, generic_constructor_input_fn),
    ("zeros", torch.zeros, generic_constructor_input_fn),
    # generic tensor-like constructor
    ("rand_like", torch.rand_like, unary_input_fn),
    ("randn_like", torch.randn_like, unary_input_fn),
    ("ones_like", torch.ones_like, unary_input_fn),
    ("zeros_like", torch.zeros_like, unary_input_fn),
    # tensor constructor with given value
    ("fill", torch.fill, fill_input_fn),
    ("full", torch.full, full_input_fn),
    ("full_like", torch.full_like, full_like_input_fn),
]


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(op, fn, input_fn, marks=getattr(pytest.mark, op, None))
        for op, fn, input_fn in tensor_constructor_operations
    ],
)
def test_tensor_constructor_benchmark(op_name, torch_op, input_fn):
    bench = GenericBenchmark(input_fn=input_fn, op_name=op_name, torch_op=torch_op)
    bench.run()
