from typing import Generator

import pytest
import torch

from flag_gems.utils import shape_utils

from .attri_util import BOOL_DTYPES, FLOAT_DTYPES, INT_DTYPES, BenchLevel
from .performance_utils import (
    Benchmark,
    Config,
    GenericBenchmark,
    GenericBenchmark2DOnly,
    SkipVersion,
    generate_tensor_input,
    unary_input_fn,
)


class UnaryReductionBenchmark(Benchmark):
    """
    Base class for benchmarking reduction operations.
    """

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = sum([shape_utils.size_in_bytes(item) for item in [inp, inp]])
        return io_amount * 1e-9 / (latency * 1e-3)

    def set_more_shapes(self):
        more_shapes_1d = [
            (1025 * 1024,),
            (1024 * 1024 * 1024,),
        ]
        more_shapes_2d = [(1024, 2**i) for i in range(0, 21, 4)]
        more_shapes_3d = [(64, 2**i, 64) for i in range(0, 15, 4)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            if inp.ndim > 1:
                yield inp, 1
            else:
                yield inp,


forward_operations = [
    ("all", torch.all, FLOAT_DTYPES),
    ("amax", torch.amax, FLOAT_DTYPES),
    ("any", torch.any, FLOAT_DTYPES),
    ("argmax", torch.argmax, FLOAT_DTYPES),
    ("argmin", torch.argmin, FLOAT_DTYPES),
    ("max", torch.max, FLOAT_DTYPES),
    ("mean", torch.mean, FLOAT_DTYPES),
    ("min", torch.min, FLOAT_DTYPES),
    ("prod", torch.prod, FLOAT_DTYPES),
    ("softmax", torch.nn.functional.softmax, FLOAT_DTYPES),
    ("sum", torch.sum, FLOAT_DTYPES),
    ("var_mean", torch.var_mean, FLOAT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(name, op, dtype, marks=getattr(pytest.mark, name, None))
        for name, op, dtype in forward_operations
    ],
)
def test_general_reduction_perf(op_name, torch_op, dtypes):
    bench = UnaryReductionBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()


backward_operations = [
    ("softmax", torch.nn.functional.softmax, FLOAT_DTYPES),
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
def test_general_reduction_backward_perf(op_name, torch_op, dtypes):
    bench = UnaryReductionBenchmark(
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
        is_backward=True,
    )
    bench.run()


def cross_entropy_loss_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    target = torch.randint(0, shape[-1], (shape[0],), device=device)
    yield inp, target
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        weight = torch.randn(shape[-1], dtype=cur_dtype, device=device)
        yield inp, target, {"weight": weight, "ignore_index": 1, "reduction": "none"}
        yield inp, target, {
            "weight": weight,
            "reduction": "sum",
            "label_smoothing": 0.1,
        }


def nll_loss_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    target = torch.randint(0, shape[-1], (shape[0],), device=device)
    yield inp, target
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        weight = torch.randn(shape[-1], dtype=cur_dtype, device=device)
        yield inp, target, {"weight": weight, "ignore_index": 1, "reduction": "none"}


def cumsum_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    yield inp, 1


def mse_loss_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    target = generate_tensor_input(shape, cur_dtype, device)
    yield inp, target
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield inp, target, {"reduction": "mean"}
        yield inp, target, {"reduction": "sum"}
        yield inp, target, {"reduction": "none"}


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "log_softmax",
            torch.nn.functional.log_softmax,
            unary_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.log_softmax,
        ),
        ### TODO:ERROR
        # pytest.param(
        #     "nonzero",
        #     torch.nonzero,
        #     unary_input_fn,
        #     FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES,
        #     marks=pytest.mark.nonzero,
        # ),
        pytest.param(
            "CrossEntropyLoss",
            torch.nn.functional.cross_entropy,
            cross_entropy_loss_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.CrossEntropyLoss,
        ),
        pytest.param(
            "cumsum",
            torch.cumsum,
            cumsum_input_fn,
            FLOAT_DTYPES + INT_DTYPES,
            marks=pytest.mark.cumsum,
        ),
        pytest.param(
            "cummin",
            torch.cummin,
            cumsum_input_fn,
            FLOAT_DTYPES + INT_DTYPES,
            marks=[
                pytest.mark.cummin,
                pytest.mark.skipif(True, reason="triton not supported"),
            ],
        ),
        pytest.param(
            "nll_loss",
            torch.nn.functional.nll_loss,
            nll_loss_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.NLLLoss,
        ),
        pytest.param(
            "mse_loss",
            torch.nn.functional.mse_loss,
            mse_loss_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.MSELoss,
        ),
    ],
)
def test_generic_reduction_benchmark(op_name, torch_op, input_fn, dtypes):
    bench = GenericBenchmark2DOnly(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=dtypes
    )
    bench.run()


@pytest.mark.count_nonzero
def test_perf_count_nonzero():
    def count_nonzero_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        dim = random.choice([None, 0, 1])

        yield inp, dim

    bench = GenericBenchmark2DOnly(
        input_fn=count_nonzero_input_fn,
        op_name="count_nonzero",
        torch_op=torch.count_nonzero,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class quantileBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        more_shapes_1d = [(4,), (1024,), (65535)]
        more_shapes_2d = [(1024, 2**i) for i in range(0, 15, 3)]
        more_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 3)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d


def quantile_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    q = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=cur_dtype, device=device)
    yield inp, q, 0


@pytest.mark.skipif(True, reason="Skipping Triton version")
@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "quantile",
            torch.quantile,
            quantile_input_fn,
            [torch.float32],
            marks=pytest.mark.quantile,
        )
    ],
)
def test_quantile_benchmark(op_name, torch_op, input_fn, dtypes):
    bench = quantileBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=dtypes
    )
    bench.run()


@pytest.mark.slice_scatter
def test_slice_scatter_perf():
    def slice_scatter_input_fn(shape, dtype, device):
        dim = random.choice([0, 1])
        start = 16
        end = 1024
        step = 2

        inp = torch.randn(shape, dtype=dtype, device=device)

        range = end - start
        valid_shape = list(inp.shape)
        if end < start:
            range = 0
        elif (end - start) > valid_shape[dim]:
            range = valid_shape[dim]
            start = 0
            end = valid_shape[dim]

        valid_shape[dim] = (range + (step - 1)) // step
        src = torch.randn(valid_shape, dtype=dtype, device=device)
        yield inp, src, dim, start, end, step

    bench = GenericBenchmark2DOnly(
        op_name="slice_scatter",
        torch_op=torch.slice_scatter,
        input_fn=slice_scatter_input_fn,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.select_scatter
def test_select_scatter_perf():
    def select_scatter_input_fn(shape, dtype, device):
        dim = random.choice([0, 1])
        index = random.randint(0, shape[dim] - 1)
        inp = torch.randn(shape, dtype=dtype, device=device)

        src_shape = list(inp.shape)
        del src_shape[dim]
        src = torch.randn(src_shape, dtype=dtype, device=device)

        yield inp, src, dim, index

    bench = GenericBenchmark2DOnly(
        op_name="select_scatter",
        torch_op=torch.select_scatter,
        input_fn=select_scatter_input_fn,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
