import random

import pytest
import torch

from flag_gems.utils import shape_utils

from .attri_util import FLOAT_DTYPES
from .performance_utils import GenericBenchmark2DOnly, generate_tensor_input


class TensorSelectBenchmark(GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        shapes = super().set_more_shapes()
        shapes = [
            # this filter is for scatter
            shape
            for shape in shapes
            if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
        ]
        return shapes


def index_select_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    threshold = 0.1
    dim = 0
    index_size = inp.size(dim)
    from math import floor

    index = torch.randint(0, index_size, [floor(index_size * threshold)], device=device)
    yield inp, dim, index


def masked_select_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    mask = generate_tensor_input(shape, cur_dtype, device) < 0.3
    yield inp, mask


def mask_select_gbps(bench_fn_args, latency):
    mask = bench_fn_args[1]
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [mask]])
    io_amount += 2 * int(torch.sum(mask))
    return io_amount * 1e-9 / (latency * 1e-3)


def index_select_gbps(bench_fn_args, latency):
    inp = bench_fn_args[0]
    dim = bench_fn_args[1]
    io_amount = shape_utils.size_in_bytes(inp) * 2 // inp.size(dim)
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, gbps_fn, dtypes",
    [
        pytest.param(
            "index_select",
            torch.index_select,
            index_select_input_fn,
            index_select_gbps,
            FLOAT_DTYPES,
            marks=pytest.mark.index_select,
        ),
        pytest.param(
            "masked_select",
            torch.masked_select,
            masked_select_input_fn,
            mask_select_gbps,
            FLOAT_DTYPES,
            marks=pytest.mark.masked_select,
        ),
    ],
)
def test_generic_reduction_benchmark(op_name, torch_op, input_fn, gbps_fn, dtypes):
    bench = TensorSelectBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
        get_gbps=gbps_fn,
    )
    bench.run()


def gather_scatter_gbps(bench_fn_args, latency):
    inp, dim, index = bench_fn_args[:3]
    data_shape = list(inp.shape)
    data_shape[dim] = index.shape[dim]
    data = torch.empty(data_shape, dtype=inp.dtype, device=inp.device)
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [index, data, data]])
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.scatter
def test_perf_scatter():
    def scatter_input_fn(shape, dtype, device):
        input_gen = gather_input_fn(shape, dtype, device)
        inp, dim, index = next(input_gen)
        src_shape = list(size + 16 for size in index.shape)
        src = torch.randn(src_shape, dtype=dtype, device=device)
        yield inp, dim, index, src

    bench = TensorSelectBenchmark(
        op_name="scatter",
        torch_op=torch.scatter,
        input_fn=scatter_input_fn,
        get_gbps=gather_scatter_gbps,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.scatter_add
def test_perf_scatter_add():
    def scatter_input_fn(shape, dtype, device):
        input_gen = gather_input_fn(shape, dtype, device)
        inp, dim, index = next(input_gen)
        src_shape = list(size + 16 for size in index.shape)
        src = torch.randn(src_shape, dtype=dtype, device=device)

        yield inp, dim, index, src, "add"

    bench = TensorSelectBenchmark(
        op_name="scatter.reduce",
        torch_op=torch.scatter,
        input_fn=scatter_input_fn,
        get_gbps=gather_scatter_gbps,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.scatter_multiply
def test_perf_scatter_multiply():
    def scatter_input_fn(shape, dtype, device):
        input_gen = gather_input_fn(shape, dtype, device)
        inp, dim, index = next(input_gen)
        src_shape = list(size + 16 for size in index.shape)
        src = torch.randn(src_shape, dtype=dtype, device=device)

        yield inp, dim, index, src, "multiply"

    bench = TensorSelectBenchmark(
        op_name="scatter.reduce",
        torch_op=torch.scatter,
        input_fn=scatter_input_fn,
        get_gbps=gather_scatter_gbps,
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()


def gather_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)

    dim = -1
    size_dim = shape[dim]
    index_shape = list(shape)
    index_shape[dim] = 2 * shape[dim]
    index = torch.randint(0, size_dim, index_shape, dtype=torch.long, device=device)
    yield inp, dim, index


@pytest.mark.gather
def test_perf_gather():
    bench = TensorSelectBenchmark(
        op_name="gather",
        torch_op=torch.gather,
        input_fn=gather_input_fn,
        get_gbps=gather_scatter_gbps,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


def slice_scatter_gbps(bench_fn_args, latency):
    inp = bench_fn_args[0]
    src = bench_fn_args[1]
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [inp, src, src]])
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.gather_backward
def test_perf_gather_backward():
    bench = TensorSelectBenchmark(
        op_name="gather_backward",
        torch_op=torch.gather,
        input_fn=gather_input_fn,
        get_gbps=gather_scatter_gbps,
        dtypes=[torch.float32],
        is_backward=True,
    )
    bench.run()


@pytest.mark.slice_scatter
def test_slice_scatter_perf():
    def slice_scatter_input_fn(shape, dtype, device):
        dim = 0 if len(shape) == 1 else 1
        start = 0
        end = shape[dim]
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

    bench = TensorSelectBenchmark(
        op_name="slice_scatter",
        torch_op=torch.slice_scatter,
        input_fn=slice_scatter_input_fn,
        dtypes=FLOAT_DTYPES,
        get_gbps=slice_scatter_gbps,
    )
    bench.run()


@pytest.mark.select_scatter
def test_select_scatter_perf():
    def select_scatter_input_fn(shape, dtype, device):
        dim = 0 if len(shape) == 1 else 1
        index = random.randint(0, shape[dim] - 1)
        inp = torch.randn(shape, dtype=dtype, device=device)

        src_shape = list(inp.shape)
        del src_shape[dim]
        src = torch.randn(src_shape, dtype=dtype, device=device)

        yield inp, src, dim, index

    bench = TensorSelectBenchmark(
        op_name="select_scatter",
        torch_op=torch.select_scatter,
        input_fn=select_scatter_input_fn,
        dtypes=FLOAT_DTYPES,
        get_gbps=slice_scatter_gbps,
    )
    bench.run()


@pytest.mark.index_add
def test_index_add_perf():
    def index_add_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device="cuda")
        dim = 0
        src_shape = list(inp.shape)
        index_max = src_shape[dim]
        index_len = index_max // 2
        index = torch.randint(0, index_max, (index_len,), device="cuda")
        src_shape[dim] = index_len
        src = torch.randn(src_shape, dtype=dtype, device="cuda")
        yield inp, dim, index, src

    bench = TensorSelectBenchmark(
        op_name="index_add",
        torch_op=torch.index_add,
        input_fn=index_add_input_fn,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
