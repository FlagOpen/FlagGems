import math
import random
from typing import Generator

import pytest
import torch

from .attri_util import (
    DEFAULT_SHAPES_2D_ONLY,
    DEFAULT_SHAPES_EXCLUDE_1D,
    DEFAULT_SHAPES_EXCLUDE_3D,
    FLOAT_DTYPES,
    INT_DTYPES,
    BenchLevel,
)
from .performance_utils import (
    Benchmark,
    Config,
    GenericBenchmark,
    GenericBenchmark2DOnly,
    GenericBenchmarkExcluse1D,
    GenericBenchmarkExcluse3D,
    generate_tensor_input,
)


def topk_input_fn(shape, dtype, device):
    x = torch.randn(shape, device=device, dtype=dtype)
    k = 5 if shape[-1] > 5 else shape[-1]
    yield {"x": x, "k": k, "dim": -1},
    # TODO:  Currently only support sorted == True and only support topk in last dimension
    # if Config.bench_level == BenchLevel.COMPREHENSIVE:
    #     k = 5 if shape[0] > 5 else shape[0]
    #     yield {"x": x, "k": k, "dim": 0},
    #     yield {"x": x, "k": k, "dim": -1, "sorted": False},


def sort_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    yield inp, {"dim": -1},
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield inp, {"dim": 0},


def resolve_neg_input_fn(shape, dtype, device):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    yield x.conj().imag,


def resolve_conj_input_fn(shape, dtype, device):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    yield x.conj(),


special_operations = [
    # Sorting Operations
    ("topk", torch.topk, FLOAT_DTYPES, topk_input_fn),
    ("sort", torch.sort, FLOAT_DTYPES, sort_input_fn),
    # Complex Operations
    ("resolve_neg", torch.resolve_neg, [torch.cfloat], resolve_neg_input_fn),
    ("resolve_conj", torch.resolve_conj, [torch.cfloat], resolve_conj_input_fn),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes, input_fn",
    [
        pytest.param(
            op,
            fn,
            dtypes,
            input_fn,
            marks=getattr(pytest.mark, op, None)(
                recommended_shapes=DEFAULT_SHAPES_EXCLUDE_1D
            ),
        )
        for op, fn, dtypes, input_fn in special_operations
    ],
)
def test_special_operations_benchmark(op_name, torch_op, dtypes, input_fn):
    bench = GenericBenchmarkExcluse1D(
        input_fn=input_fn, op_name=op_name, dtypes=dtypes, torch_op=torch_op
    )
    bench.run()


@pytest.mark.isin(recommended_shapes=DEFAULT_SHAPES_2D_ONLY)
def test_isin_perf():
    def isin_input_fn(shape, dtype, device):
        elements = generate_tensor_input(shape, dtype, device)
        test_elements = generate_tensor_input(shape, dtype, device)
        yield elements, test_elements
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            # assume_unique set to True
            uniq_elements = torch.unique(generate_tensor_input(shape, dtype, device))
            uniq_test_elements = torch.unique(
                generate_tensor_input(shape, dtype, device)
            )
            yield uniq_elements, uniq_test_elements, {"assume_unique": True}

    bench = GenericBenchmark2DOnly(
        input_fn=isin_input_fn,
        op_name="isin",
        torch_op=torch.isin,
        dtypes=FLOAT_DTYPES + INT_DTYPES,
    )
    bench.run()


@pytest.mark.unique(recommended_shapes=DEFAULT_SHAPES_2D_ONLY)
def test_perf_unique():
    def unique_input_fn(shape, dtype, device):
        inp = generate_tensor_input(shape, dtype, device)
        yield inp, {"sorted": True, "return_inverse": True, "return_counts": False},
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            yield inp, {"sorted": True, "return_inverse": False, "return_counts": True},

    bench = GenericBenchmark2DOnly(
        input_fn=unique_input_fn,
        op_name="unique",
        torch_op=torch.unique,
        dtypes=INT_DTYPES,
    )
    bench.run()


@pytest.mark.multinomial(recommended_shapes=DEFAULT_SHAPES_2D_ONLY)
def test_multinomial_with_replacement():
    def multinomial_input_fn(shape, dtype, device):
        dist = torch.rand(shape, dtype=dtype, device=device)
        n_samples = 10000
        yield dist, n_samples, True,

    bench = GenericBenchmark2DOnly(
        input_fn=multinomial_input_fn,
        op_name="multinomial",
        torch_op=torch.multinomial,
        dtypes=(torch.float16, torch.float32),
    )
    bench.run()


@pytest.mark.pad
def test_perf_pad():
    def padding_input_fn(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        rank = input.ndim
        pad_params = [random.randint(0, 10) for _ in range(rank * 2)]
        pad_value = float(torch.randint(0, 1024, [1]))
        yield input, {
            "pad": pad_params,
            "mode": "constant",
            "value": pad_value,
        },

    bench = GenericBenchmark(
        input_fn=padding_input_fn,
        op_name="padding",
        torch_op=torch.nn.functional.pad,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


# TODO:add 3d shapes
EMBEDDING_RECOMMENDED_SHAPES = [
    (4,  4),
    (16, 16),
    (128, 128),
    (256, 256),
    (1024, 1024),
]

class EmbeddingBenchmark(GenericBenchmark2DOnly):
    DEFAULT_SHAPES = EMBEDDING_RECOMMENDED_SHAPES
    def set_shapes(self):
        self.shapes = self.DEFAULT_SHAPES
        #TODO: add more shapes

@pytest.mark.embedding(recommended_shapes=DEFAULT_SHAPES_2D_ONLY)
def test_perf_embedding():
    def embedding_input_fn(shape, dtype, device):
            num_embeddings, embedding_dim = shape
            indices = torch.randint(
                0, num_embeddings, (num_embeddings,), device=device
            )
            weight = torch.randn(
                (num_embeddings, embedding_dim), device=device, dtype=dtype
            )
            yield {"input": indices, "weight": weight},
            if Config.bench_level == BenchLevel.COMPREHENSIVE:
                indices_2d = torch.randint(
                    0,
                    num_embeddings,
                    (num_embeddings, num_embeddings),
                    device=device,
                )
                yield {"input": indices_2d, "weight": weight},
    bench = EmbeddingBenchmark(
        input_fn=embedding_input_fn,
        op_name="embedding",
        torch_op=torch.nn.functional.embedding,
        dtypes=[
            torch.float32,
            torch.float16,
        ],  # Note(Zhengzekang): triton do not support bfloat16 atomic add which is used in embedding grad.
    )
    bench.run()


# [N, C, H, W]
UPSAMPLE_SHAPES = [
    (1, 3, 512, 512),
    (8, 16, 128, 128),
    (2, 3, 1024, 1024),
    (16, 16, 512, 512),
    (16, 16, 1024, 1024),
]


class UpsampleBenchmark(Benchmark):
    DEFAULT_SHAPES = UPSAMPLE_SHAPES

    def set_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        self.shapes = self.DEFAULT_SHAPES[:]
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            more_shapes = []
            # TODO: more shapes
            self.shapes.extend(more_shapes)

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            batch, channel, height, weight = shape
            input = torch.randn(size=shape, device=self.device, dtype=cur_dtype)
            scale_factors = (2, 2)
            output_size = (
                int(height * scale_factors[0]),
                int(weight * scale_factors[1]),
            )
            yield {
                "input": input,
                "output_size": output_size,
                "align_corners": False,
                "scales_h": None,
                "scales_w": None,
            },


@pytest.mark.upsample_bicubic2d_aa(
    recommended_shapes=UPSAMPLE_SHAPES, shape_desc="N, C, H, W"
)
def test_perf_upsample_bicubic2d_aa():
    bench = UpsampleBenchmark(
        op_name="upsample_bicubic2d_aa",
        torch_op=torch._C._nn._upsample_bicubic2d_aa,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
