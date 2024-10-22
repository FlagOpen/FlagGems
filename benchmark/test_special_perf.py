import math
import random
from typing import Generator

import pytest
import torch

from .attri_util import FLOAT_DTYPES, INT_DTYPES, BenchLevel
from .performance_utils import (
    Benchmark,
    Config,
    GenericBenchmark,
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


# TODO: set shape for isin. meet CUDA out of memory
def isin_input_fn(shape, dtype, device):
    elements = generate_tensor_input(shape, dtype, device)
    test_elements = generate_tensor_input(shape, dtype, device)
    yield elements, test_elements
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        # assume_unique set to True
        uniq_elements = torch.unique(generate_tensor_input(shape, dtype, device))
        uniq_test_elements = torch.unique(generate_tensor_input(shape, dtype, device))
        yield uniq_elements, uniq_test_elements, {"assume_unique": True}


special_operations = [
    # Sorting Operations
    ("topk", torch.topk, FLOAT_DTYPES, topk_input_fn),
    ("sort", torch.sort, FLOAT_DTYPES, sort_input_fn),
    # Complex Operations
    ("resolve_neg", torch.resolve_neg, [torch.cfloat], resolve_neg_input_fn),
    ("resolve_conj", torch.resolve_conj, [torch.cfloat], resolve_conj_input_fn),
    # Numerical Check
    ("isin", torch.isin, INT_DTYPES + FLOAT_DTYPES, isin_input_fn),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes, input_fn",
    [
        pytest.param(op, fn, dtypes, input_fn, marks=getattr(pytest.mark, op, None))
        for op, fn, dtypes, input_fn in special_operations
    ],
)
def test_special_operations_benchmark(op_name, torch_op, dtypes, input_fn):
    bench = GenericBenchmark(
        input_fn=input_fn, op_name=op_name, dtypes=dtypes, torch_op=torch_op
    )
    bench.run()


@pytest.mark.unique
def test_perf_unique():
    def unique_input_fn(shape, dtype, device):
        inp = generate_tensor_input(shape, dtype, device)
        yield inp, {"sorted": True, "return_inverse": True, "return_counts": False},
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            yield inp, {"sorted": True, "return_inverse": False, "return_counts": True},

    bench = GenericBenchmark(
        input_fn=unique_input_fn,
        op_name="unique",
        torch_op=torch.unique,
        dtypes=INT_DTYPES,
    )
    bench.run()


# TODO: not supported for 3D. prob_dist must be 1 or 2 dim.
@pytest.mark.multinomial
def test_multinomial_with_replacement():
    def multinomial_input_fn(shape, dtype, device):
        dist = torch.rand(shape, dtype=dtype, device=device)
        n_samples = 10000
        yield dist, n_samples, True,

    bench = GenericBenchmark(
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


# TODO: special case for tensor construct, move to other file
@pytest.mark.arange
def test_perf_arange():
    def arange_input_fn(shape, dtype, device):
        yield {
            "end": math.prod(shape),
            "device": device,
            "dtype": dtype,
        },
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            yield {
                "start": 0,
                "end": math.prod(shape),
                "step": 2,
                "device": device,
                "dtype": dtype,
            },

    bench = GenericBenchmark(
        input_fn=arange_input_fn,
        op_name="arange",
        torch_op=torch.arange,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.repeat_interleave
def test_perf_repeat_interleave():
    def repeat_interleave_self_input_fn(shape, dtype, device):
        if dtype in FLOAT_DTYPES:
            # torch.repeat_interleave(input, repeats, dim=None, *, output_size=None) → Tensor
            inp = torch.randn(shape, dtype=dtype, device=device)
            repeats = 3
            yield inp, repeats
        elif dtype == torch.int32:
            # torch.repeat_interleave(repeats, *) → Tensor
            repeats = torch.randint(
                low=0,
                high=0x7F,
                size=[
                    shape[-1],
                ],
                dtype=dtype,
                device=device,
            )
            yield repeats,

    bench = GenericBenchmark(
        op_name="repeat_interleave",
        torch_op=torch.repeat_interleave,
        input_fn=repeat_interleave_self_input_fn,
        dtypes=FLOAT_DTYPES + [torch.int32],
    )
    bench.run()


# TODO: add more support for embedding
EMBEDDING_RECOMMENDED_SHAPES = [
    (2, 4, 128),
    (2, 8, 256),
    (4, 4, 128),
    (4, 8, 256),
    (4, 8, 4096),
]


class EmbeddingBenchmark(Benchmark):
    # DEFAULT_SHAPES = EMBEDDING_RECOMMENDED_SHAPES
    def set_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (B, M, N).
        self.shapes = self.DEFAULT_SHAPES[:]
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            more_shapes = []
            # TODO: more shapes
            self.shapes.extend(more_shapes)

    def get_input_iter(self, cur_dtype) -> Generator:
        for num_embeddings, embedding_dim in self.shapes:
            indices = torch.randint(
                0, num_embeddings, (num_embeddings,), device=self.device
            )
            weight = torch.randn(
                (num_embeddings, embedding_dim), device=self.device, dtype=cur_dtype
            )
            yield {"input": indices, "weight": weight},
            if Config.bench_level == BenchLevel.COMPREHENSIVE:
                indices_2d = torch.randint(
                    0,
                    num_embeddings,
                    (num_embeddings, num_embeddings),
                    device=self.device,
                )
                yield {"input": indices_2d, "weight": weight},


# TODO: check the recommeded shapes again
# TODO: add more benchmark shapes when comprehensive situation
@pytest.mark.embedding()
def test_perf_embedding():
    bench = EmbeddingBenchmark(
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
        op_name="_upsample_bicubic2d_aa",
        torch_op=torch._C._nn._upsample_bicubic2d_aa,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
