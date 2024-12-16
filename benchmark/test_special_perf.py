import random

import pytest
import torch

from .attri_util import BOOL_DTYPES, FLOAT_DTYPES, INT_DTYPES, BenchLevel
from .performance_utils import (
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


def resolve_neg_input_fn(shape, dtype, device):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    yield x.conj().imag,


def resolve_conj_input_fn(shape, dtype, device):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    yield x.conj(),


special_operations = [
    # Sorting Operations
    ("topk", torch.topk, FLOAT_DTYPES, topk_input_fn),
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
            marks=getattr(pytest.mark, op, None),
        )
        for op, fn, dtypes, input_fn in special_operations
    ],
)
def test_special_operations_benchmark(op_name, torch_op, dtypes, input_fn):
    bench = GenericBenchmarkExcluse1D(
        input_fn=input_fn, op_name=op_name, dtypes=dtypes, torch_op=torch_op
    )
    bench.run()


@pytest.mark.isin
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
        dtypes=INT_DTYPES,
    )
    bench.run()


@pytest.mark.unique
def test_perf_unique():
    def unique_input_fn(shape, dtype, device):
        inp = generate_tensor_input(shape, dtype, device)
        yield inp, {"sorted": True, "return_inverse": True, "return_counts": False},

    bench = GenericBenchmark2DOnly(
        input_fn=unique_input_fn,
        op_name="unique",
        torch_op=torch.unique,
        dtypes=INT_DTYPES,
    )
    bench.run()


@pytest.mark.sort
def test_perf_sort():
    class SortBenchmark(GenericBenchmark2DOnly):
        def set_more_shapes(self):
            return [(1024, 1), (1024, 512)]

    def sort_input_fn(shape, dtype, device):
        inp = generate_tensor_input(shape, dtype, device)
        yield inp, {"dim": -1, "descending": False},

    bench = SortBenchmark(
        input_fn=sort_input_fn,
        op_name="sort",
        torch_op=torch.sort,
        dtypes=INT_DTYPES + FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.multinomial
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


class EmbeddingBenchmark(GenericBenchmark2DOnly):
    def set_more_shapes(self):
        # TODO: add more shapes
        return None


@pytest.mark.embedding
def test_perf_embedding():
    def embedding_input_fn(shape, dtype, device):
        num_embeddings, embedding_dim = shape
        indices = torch.randint(0, num_embeddings, (num_embeddings,), device=device)
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


class UpsampleBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return None


@pytest.mark.upsample_bicubic2d_aa
def test_perf_upsample_bicubic2d_aa():
    def upsample_bicubic2d_aa_input_fn(shape, dtype, device):
        batch, channel, height, weight = shape
        input = torch.randn(size=shape, device=device, dtype=dtype)
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

    bench = UpsampleBenchmark(
        input_fn=upsample_bicubic2d_aa_input_fn,
        op_name="upsample_bicubic2d_aa",
        torch_op=torch._C._nn._upsample_bicubic2d_aa,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.upsample_nearest2d
def test_perf_upsample_nearest2d():
    def upsample_nearest2d_input_fn(shape, dtype, device):
        batch, channel, height, weight = shape
        input = torch.randn(size=shape, device=device, dtype=dtype)
        scale_factors = (2, 2)
        output_size = (
            int(height * scale_factors[0]),
            int(weight * scale_factors[1]),
        )
        yield {
            "input": input,
            "output_size": output_size,
            "scales_h": None,
            "scales_w": None,
        },

    bench = UpsampleBenchmark(
        input_fn=upsample_nearest2d_input_fn,
        op_name="upsample_nearest2d",
        torch_op=torch._C._nn.upsample_nearest2d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class ConvBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return None


@pytest.mark.conv2d
def test_perf_conv2d():
    def conv2d_input_fn(shape, dtype, device):
        (
            batch,
            input_c,
            input_h,
            input_w,
            out_c,
            kernel_h,
            kernel_w,
            stride,
            padding,
            groups,
        ) = shape
        input_shape = (batch, input_c, input_h, input_w)
        weight_shape = (out_c, input_c // groups, kernel_h, kernel_w)
        input = torch.randn(size=input_shape, device=device, dtype=dtype)

        weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

        yield {
            "input": input,
            "weight": weight,
            "bias": None,
            "groups": groups,
            "stride": stride,
            "padding": padding,
        },

    torch.backends.cudnn.allow_tf32 = False
    bench = ConvBenchmark(
        input_fn=conv2d_input_fn,
        op_name="conv2d",
        torch_op=torch.nn.functional.conv2d,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.diag
def test_perf_diag():
    def diag_input_fn(shape, dtype, device):
        input = generate_tensor_input(shape, dtype, device)
        diagonal = random.randint(-4, 4)
        yield input, {
            "diagonal": diagonal,
        },

    bench = GenericBenchmarkExcluse3D(
        input_fn=diag_input_fn,
        op_name="diag",
        torch_op=torch.diag,
        dtypes=FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.diag_embed
def test_perf_diag_embed():
    def diag_embed_input_fn(shape, dtype, device):
        inp = generate_tensor_input(shape, dtype, device)
        yield {"input": inp},

        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            yield {"input": inp, "offset": 1, "dim1": 0, "dim2": -1},

    bench = EmbeddingBenchmark(
        input_fn=diag_embed_input_fn,
        op_name="diag_embed",
        torch_op=torch.diag_embed,
        dtypes=FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES,
    )

    bench.run()


@pytest.mark.diagonal_backward
def test_perf_diagonal_backward():
    def diagonal_backward_input_fn(shape, dtype, device):
        inp = generate_tensor_input(shape, dtype, device)
        yield inp,

        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            yield inp, {"offset": 1, "dim1": 0, "dim2": -1},

    bench = GenericBenchmarkExcluse1D(
        input_fn=diagonal_backward_input_fn,
        op_name="diagonal_backward",
        torch_op=torch.diagonal,
        dtypes=FLOAT_DTYPES,
        is_backward=True,
    )

    bench.run()
