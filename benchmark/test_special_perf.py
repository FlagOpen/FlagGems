import math

import pytest
import torch

from .attri_util import (
    DEFAULT_BATCH,
    DEFAULT_NON_BLAS_BENCH_SHAPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    BenchLevel,
)
from .performance_utils import Benchmark, Config, binary_args, unary_arg


# TODO: enable @pytest.mark.native_dropout or not
@pytest.mark.dropout
def test_perf_dropout():
    bench = Benchmark(
        op_name="dropout",
        torch_op=torch.nn.Dropout(p=0.5),
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
    )
    bench.run()


EMBEDDING_SIZE = 1024
EMBEDDING_RECOMMENDED_SHAPES = [  # [B, M, N] mode
    (2, 4, 128),
    (2, 8, 256),
    (4, 4, 128),
    (4, 8, 256),
    (4, 8, 4096),
]


# TODO: check the recommeded shapes again
# TODO: add more benchmark shapes when comprehensive situation
@pytest.mark.embedding(recommended_shapes=EMBEDDING_RECOMMENDED_SHAPES)
def test_perf_embedding():
    def embedding_kwargs(dtype, embedding_size, shape):
        input = torch.randint(0, embedding_size, (shape[0], shape[1]), device="cuda")
        weight = torch.randn((embedding_size, shape[2]), device="cuda", dtype=dtype)
        return {"input": input, "weight": weight}

    bench = Benchmark(
        op_name="embedding",
        torch_op=torch.nn.functional.embedding,
        arg_func=None,
        dtypes=[
            torch.float32,
            torch.float16,
        ],  # Note(Zhengzekang): triton do not support bfloat16 atomic add which is used in embedding grad.
        batch=EMBEDDING_SIZE,
        sizes=EMBEDDING_RECOMMENDED_SHAPES,
        kwargs_func=embedding_kwargs,
    )
    bench.run()


@pytest.mark.topk
def test_perf_topk():
    def topk_kwargs(dtype, batch, shape):
        x = torch.randn(shape, device="cuda", dtype=dtype)
        return {"x": x, "k": 5, "dim": -1}

    bench = Benchmark(
        op_name="topk",
        torch_op=torch.topk,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
        kwargs_func=topk_kwargs,
    )
    bench.run()


@pytest.mark.resolve_neg
def test_perf_resolve_neg():
    def resolve_neg_arg(dtype, batch, shape):
        x = torch.randn(size=shape, dtype=dtype, device="cuda")
        y = x.conj()
        z = y.imag
        return (z,)

    bench = Benchmark(
        op_name="resolve_neg",
        torch_op=torch.resolve_neg,
        arg_func=resolve_neg_arg,
        dtypes=[torch.cfloat],
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
    )
    bench.run()


@pytest.mark.resolve_conj
def test_perf_resolve_conj():
    def resolve_conj_arg(dtype, batch, shape):
        x = torch.randn(size=shape, dtype=dtype, device="cuda")
        return (x.conj(),)

    bench = Benchmark(
        op_name="resolve_conj",
        torch_op=torch.resolve_conj,
        arg_func=resolve_conj_arg,
        dtypes=[torch.cfloat],
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
    )
    bench.run()


@pytest.mark.unique
def test_perf_unique():
    def unique_kwargs(dtype, batch, shape):
        return {"sorted": True, "return_inverse": True, "return_counts": False}

    bench = Benchmark(
        op_name="unique",
        torch_op=torch.unique,
        arg_func=unary_arg,
        dtypes=INT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
        kwargs_func=unique_kwargs,
    )
    bench.run()


MULTINOMIAL_SHAPES = DEFAULT_NON_BLAS_BENCH_SHAPES[:]
if Config.bench_level == BenchLevel.COMPREHENSIVE:
    MULTINOMIAL_SHAPES.extend(
        [
            (1,),
            (16,),
            (64,),
            (256,),
            (1024,),
            (33,),
            (81,),
            (273,),
            (1041,),
            (1, 1),
            (1, 16),
            (1, 64),
            (1, 1000),
            (5, 1),
            (5, 16),
            (5, 64),
            (5, 1000),
            (1024, 1),
            (1024, 16),
            (1024, 1000),
        ]
    )


@pytest.mark.multinomial
def test_multinomial_with_replacement():
    def multinomial_args(dtype, batch, shape):
        dist = torch.rand(shape, dtype=dtype, device="cuda")
        n_samples = 10000
        return (dist, n_samples, True)

    bench = Benchmark(
        op_name="multinomial",
        torch_op=torch.multinomial,
        arg_func=multinomial_args,
        dtypes=(torch.float16, torch.float32),
        batch=DEFAULT_BATCH,
        sizes=MULTINOMIAL_SHAPES,
    )
    bench.run()


PAD_SHAPES = DEFAULT_NON_BLAS_BENCH_SHAPES[:]
if Config.bench_level == BenchLevel.COMPREHENSIVE:
    PAD_SHAPES.extend(
        [
            (64, 64, 64, 64),
        ]
    )


@pytest.mark.pad
def test_perf_pad():
    def padding_kwargs(dtype, batch, shape):
        input = torch.randn(shape, device="cuda", dtype=dtype)
        rank = input.ndim
        pad_params = tuple(torch.randint(0, 10, [rank * 2]))
        pad_value = float(torch.randint(0, 1024, [1]))
        return {
            "input": input,
            "pad": pad_params,
            "mode": "constant",
            "value": pad_value,
        }

    bench = Benchmark(
        op_name="padding",
        torch_op=torch.nn.functional.pad,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=PAD_SHAPES,
        kwargs_func=padding_kwargs,
    )
    bench.run()


# TODO: Add more benchmark shapes to handle three types of parameters: start, step, and end.
@pytest.mark.arange
def test_perf_arange():
    def arange_kwargs(dtype, batch, shape):
        return {
            "end": math.prod(shape),
            "device": "cuda",
            "dtype": dtype,
        }

    bench = Benchmark(
        op_name="arange",
        torch_op=torch.arange,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
        kwargs_func=arange_kwargs,
    )
    bench.run()


@pytest.mark.isin
def test_perf_isin():
    bench = Benchmark(
        op_name="isin",
        torch_op=torch.isin,
        arg_func=binary_args,
        dtypes=INT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
    )
    bench.run()


@pytest.mark.fill
def test_perf_fill():
    def fill_kwargs(dtype, batch, shape):
        value = 1.0
        input = torch.empty(math.prod(shape), dtype=dtype, device="cuda")
        return {
            "input": input,
            "value": value,
        }

    bench = Benchmark(
        op_name="fill",
        torch_op=torch.fill,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
        kwargs_func=fill_kwargs,
    )
    bench.run()


STACK_AND_HSTACK_RECOMMENDED_SHAPES = [
    (512, 64),
    (512, 384),
    (512, 704),
    (512, 1024),
    (512, 1344),
]


@pytest.mark.stack(recommended_shapes=STACK_AND_HSTACK_RECOMMENDED_SHAPES)
def test_perf_stack():
    def stack_args(dtype, batch, shape):
        inp = torch.randn(size=shape, dtype=dtype, device="cuda")
        return {(inp,) * 3}

    bench = Benchmark(
        op_name="stack",
        torch_op=torch.stack,
        arg_func=stack_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=STACK_AND_HSTACK_RECOMMENDED_SHAPES,
    )
    bench.run()


@pytest.mark.hstack(recommended_shapes=STACK_AND_HSTACK_RECOMMENDED_SHAPES)
def test_perf_hstack():
    def hstack_args(dtype, batch, shape):
        inp = torch.randn(size=shape, dtype=dtype, device="cuda")
        return {(inp,) * 3}

    bench = Benchmark(
        op_name="hstack",
        torch_op=torch.hstack,
        arg_func=hstack_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=STACK_AND_HSTACK_RECOMMENDED_SHAPES,
    )
    bench.run()


CAT_SHAPES = DEFAULT_NON_BLAS_BENCH_SHAPES[:]
if Config.bench_level == BenchLevel.COMPREHENSIVE:
    CAT_SHAPES.extend(
        [
            (16, 128, 64, 64),
            # TODO: add more shapes.
        ]
    )


@pytest.mark.cat
def test_perf_cat():
    def cat_args(dtype, batch, shape):
        if dtype in FLOAT_DTYPES:
            inp1 = torch.randn(shape, dtype=dtype, device="cuda")
            inp2 = torch.randn(shape, dtype=dtype, device="cuda")
        elif dtype in INT_DTYPES:
            inp1 = torch.randint(
                low=0, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
            )
            inp2 = torch.randint(
                low=0, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
            )
        return [[inp1, inp2]]

    def cat_kwargs(dtype, batch, size):
        return {"dim": 0}

    bench = Benchmark(
        op_name="cat",
        torch_op=torch.cat,
        arg_func=cat_args,
        dtypes=FLOAT_DTYPES + INT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=CAT_SHAPES,
        kwargs_func=cat_kwargs,
    )
    bench.run()


VSTACK_RECOMMENDED_SHAPES = [
    [
        (512, 64),
        (513, 64),
        (514, 64),
    ],
    [
        (512, 384),
        (513, 384),
        (514, 384),
    ],
    [
        (512, 704),
        (513, 704),
        (514, 704),
    ],
    [
        (512, 1024),
        (513, 1024),
        (514, 1024),
    ],
    [
        (512, 1344),
        (513, 1344),
        (514, 1344),
    ],
]
VSTACK_SHAPES = VSTACK_RECOMMENDED_SHAPES[:]
if Config.bench_level == BenchLevel.COMPREHENSIVE:
    VSTACK_SHAPES.extend(
        [
            [
                (13, 3, 333),
                (17, 3, 333),
                (7, 3, 333),
            ],
            [
                (13, 3, 64, 5, 2),
                (16, 3, 64, 5, 2),
                (7, 3, 64, 5, 2),
            ],
        ]
    )


@pytest.mark.vstack(recommended_shapes=VSTACK_RECOMMENDED_SHAPES)
def test_perf_vstack():
    def vstack_args(dtype, batch, shape_list):
        inp1 = torch.randn(size=shape_list[0], dtype=dtype, device="cuda")
        inp2 = torch.randn(size=shape_list[1], dtype=dtype, device="cuda")
        inp3 = torch.randn(size=shape_list[2], dtype=dtype, device="cuda")
        return [[inp1, inp2, inp3]]

    bench = Benchmark(
        op_name="vstack",
        torch_op=torch.vstack,
        arg_func=vstack_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=VSTACK_SHAPES,
    )
    bench.run()


@pytest.mark.repeat_interleave
def test_perf_repeat_interleave():
    def repeat_interleave_self_int_arg(dtype, batch, shape):
        if dtype in FLOAT_DTYPES:
            inp = torch.randn(shape, dtype=dtype, device="cuda")
            repeats = 2
            return inp, repeats
        elif dtype == torch.int32:
            repeats = torch.randint(
                low=0,
                high=0x7F,
                size=[
                    shape[-1],
                ],
                dtype=dtype,
                device="cuda",
            )
            return (repeats,)

    bench = Benchmark(
        op_name="repeat_interleave",
        torch_op=torch.repeat_interleave,
        arg_func=repeat_interleave_self_int_arg,
        dtypes=FLOAT_DTYPES + [torch.int32],
        batch=DEFAULT_BATCH,
        sizes=DEFAULT_NON_BLAS_BENCH_SHAPES,
    )
    bench.run()
