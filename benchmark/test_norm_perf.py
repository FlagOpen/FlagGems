from typing import Generator

import pytest
import torch

from .attri_util import DEFAULT_GROUPNORM_SHAPES, BenchLevel
from .conftest import Config
from .performance_utils import Benchmark, GenericBenchmark, unary_input_fn


class GroupNormBenchmark(Benchmark):
    DEFAULT_SHAPES = DEFAULT_GROUPNORM_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        for n, c, h, w, num_groups in self.shapes:
            inp1 = torch.randn([n, c, h * w], dtype=cur_dtype, device=self.device)
            inp2 = torch.randn([n, c, h, w], dtype=cur_dtype, device=self.device)
            weight = torch.randn(
                [
                    c,
                ],
                dtype=cur_dtype,
                device=self.device,
            )
            bias = torch.randn(
                [
                    c,
                ],
                dtype=cur_dtype,
                device=self.device,
            )
            yield inp1, num_groups, weight, bias
            if Config.bench_level == BenchLevel.COMPREHENSIVE:
                yield inp2, num_groups, weight, bias

    def set_shapes(self):
        # self.shapes is a list of tuples, each containing five elements:
        # (N, C, H, W, num_groups).
        self.shapes = self.DEFAULT_SHAPES[:]
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            more_shapes = []
            # TODO: more shapes
            self.shapes.extend(more_shapes)


@pytest.mark.group_norm(recommended_shapes=DEFAULT_GROUPNORM_SHAPES)
def test_perf_group_norm():
    bench = GroupNormBenchmark(
        op_name="group_norm",
        torch_op=torch.nn.functional.group_norm,
    )
    bench.run()


# TODO: add 3D or 4D shapes for layernorm
def layernorm_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    layer_shape = shape[1:]
    weight = torch.randn(layer_shape, dtype=dtype, device=device)
    bias = torch.randn(layer_shape, dtype=dtype, device=device)
    yield inp, layer_shape, weight, bias


def weight_norm_input_fn(shape, dtype, device):
    dim = 0
    v = torch.randn(shape, dtype=dtype, device=device)
    g = torch.randn(shape[dim], dtype=dtype, device=device)
    yield v, g, dim


norm_operations = [
    ("layer_norm", torch.layer_norm, layernorm_input_fn),
    ("weight_norm_interface", torch._weight_norm_interface, weight_norm_input_fn),
    ("vector_norm", torch.linalg.vector_norm, unary_input_fn),
]


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(op, fn, input_fn, marks=getattr(pytest.mark, op, None))
        for op, fn, input_fn in norm_operations
    ],
)
def test_norm_benchmark(op_name, torch_op, input_fn):
    bench = GenericBenchmark(input_fn=input_fn, op_name=op_name, torch_op=torch_op)
    bench.run()
