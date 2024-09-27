import pytest
import torch

from .conftest import Config
from .attri_util import (
    BLAS_BATCH,
    DEFAULT_BATCH,
    FLOAT_DTYPES,
    BenchLevel,
)
from .performance_utils import Benchmark

if Config.bench_level == BenchLevel.COMPREHENSIVE:
    BLAS_MN_SHAPES = [(1, 32), (160, 1024), (5333, 497)]
    BLAS_MNK_SHAPES = [(1, 1, 32), (15, 160, 1024), (495, 5333, 71)]
    BLAS_BATCHS = [1, 4, 8, 16, 32]
elif Config.bench_level == BenchLevel.CORE:
    BLAS_MN_SHAPES = [(1, 32), (160, 1024), (5333, 497)]
    BLAS_MNK_SHAPES = [(1, 1, 32), (15, 160, 1024), (495, 5333, 71)]
    BLAS_BATCHS = [4]


def M(mnk):
    return mnk[0]


def N(mnk):
    return mnk[1]


def K(mnk):
    return mnk[2]


# the size for every  blas Operator is (m, n, k)
@pytest.mark.addmm(recommended_shapes=BLAS_MNK_SHAPES)
def test_perf_addmm():
    def addmm_args(dtype, batch, size):
        bias = torch.randn(
            [M(size), N(size)],
            dtype=dtype,
            device="cuda",
        )
        inp1 = torch.randn([M(size), K(size)], dtype=dtype, device="cuda")
        inp2 = torch.randn([K(size), N(size)], dtype=dtype, device="cuda")
        return bias, inp1, inp2

    bench = Benchmark(
        op_name="addmm",
        torch_op=torch.addmm,
        arg_func=addmm_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=BLAS_MNK_SHAPES,
    )
    bench.run()


@pytest.mark.bmm(recommended_shapes=BLAS_MNK_SHAPES)
def test_perf_bmm():
    def bmm_args(dtype, batch, size):
        inp1 = torch.randn([batch, M(size), K(size)], dtype=dtype, device="cuda")
        inp2 = torch.randn([batch, K(size), N(size)], dtype=dtype, device="cuda")
        return inp1, inp2

    bench = Benchmark(
        op_name="bmm",
        torch_op=torch.bmm,
        arg_func=bmm_args,
        dtypes=FLOAT_DTYPES,
        batch=BLAS_BATCH,
        sizes=BLAS_MNK_SHAPES,
    )
    return bench.run()


def test_perf_mm():
    def mm_args(dtype, batch, size):
        inp1 = torch.randn([M(size), K(size)], dtype=dtype, device="cuda")
        inp2 = torch.randn([K(size), N(size)], dtype=dtype, device="cuda")
        return inp1, inp2

    bench = Benchmark(
        op_name="mm",
        torch_op=torch.mm,
        arg_func=mm_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=BLAS_MNK_SHAPES,
    )
    bench.run()


def test_perf_mv():
    def mv_args(dtype, batch, size):
        inp1 = torch.randn([size[0], size[1]], dtype=dtype, device="cuda")
        inp2 = torch.randn([size[1]], dtype=dtype, device="cuda")
        return inp1, inp2

    bench = Benchmark(
        op_name="mv",
        torch_op=torch.mv,
        arg_func=mv_args,
        dtypes=FLOAT_DTYPES,
        batch=BLAS_BATCH,
        sizes=BLAS_MN_SHAPES,
    )
    bench.run()


def test_perf_outer():
    def outer_args(dtype, batch, size):
        inp1 = torch.randn(size[0], dtype=dtype, device="cuda")
        inp2 = torch.randn(size[1], dtype=dtype, device="cuda")
        return inp1, inp2

    bench = Benchmark(
        op_name="outer",
        torch_op=torch.outer,
        arg_func=outer_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=BLAS_MN_SHAPES,
    )
    bench.run()
