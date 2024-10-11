import math
from typing import Generator

import pytest
import torch

from .attri_util import FLOAT_DTYPES, INT_DTYPES, BenchLevel
from .performance_utils import (
    Benchmark,
    Config,
    GenericBenchmark,
    binary_input_fn,
    unary_input_fn,
)


def topk_input_fn(shape, dtype, device):
    x = torch.randn(shape, device=device, dtype=dtype)
    yield {"x": x, "k": 5, "dim": -1},


def resolve_neg_input_fn(shape, dtype, device):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    yield x.conj().imag,


def resolve_conj_input_fn(shape, dtype, device):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    yield x.conj(),


# Define operations and their corresponding input functions
special_operations = [
    # TODO: enable @pytest.mark.native_dropout or not
    ("dropout", torch.nn.Dropout(p=0.5), FLOAT_DTYPES, unary_input_fn),
    # TODO: comprehensive situation, size should bigger than 5.
    ("topk", torch.topk, FLOAT_DTYPES, topk_input_fn),
    ("resolve_neg", torch.resolve_neg, [torch.cfloat], resolve_neg_input_fn),
    ("resolve_conj", torch.resolve_conj, [torch.cfloat], resolve_conj_input_fn),
    ("isin", torch.isin, INT_DTYPES, binary_input_fn),
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
        inp = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device=device,
        )
        yield inp, {"sorted": True, "return_inverse": True, "return_counts": False},

    bench = GenericBenchmark(
        input_fn=unique_input_fn,
        op_name="unique",
        torch_op=torch.unique,
        dtypes=INT_DTYPES,
    )
    bench.run()


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


# PAD_SHAPES = DEFAULT_NON_BLAS_BENCH_SHAPES[:]
# if Config.bench_level == BenchLevel.COMPREHENSIVE:
#     PAD_SHAPES.extend(
#         [
#             (64, 64, 64, 64),
#         ]
#     )
@pytest.mark.pad
def test_perf_pad():
    def padding_input_fn(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        rank = input.ndim
        pad_params = tuple(torch.randint(0, 10, [rank * 2]))
        pad_value = float(torch.randint(0, 1024, [1]))
        yield {
            "input": input,
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


# # TODO: Add more GenericBenchmark shapes to handle three types of parameters: start, step, and end.
@pytest.mark.arange
def test_perf_arange():
    def arange_input_fn(shape, dtype, device):
        yield {
            "end": math.prod(shape),
            "device": "cuda",
            "dtype": dtype,
        }

    bench = GenericBenchmark(
        input_fn=arange_input_fn,
        op_name="arange",
        torch_op=torch.arange,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.fill
def test_perf_fill():
    def fill_input_fn(shape, dtype, device):
        value = 1.0
        input = torch.empty(math.prod(shape), dtype=dtype, device=device)
        yield {
            "input": input,
            "value": value,
        },

    bench = GenericBenchmark(
        input_fn=fill_input_fn,
        op_name="fill",
        torch_op=torch.fill,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.repeat_interleave
def test_perf_repeat_interleave():
    def repeat_interleave_self_input_fn(shape, dtype, device):
        if dtype in FLOAT_DTYPES:
            inp = torch.randn(shape, dtype=dtype, device=device)
            repeats = 2
            yield inp, repeats
        elif dtype == torch.int32:
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
