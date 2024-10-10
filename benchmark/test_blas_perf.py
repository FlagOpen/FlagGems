import itertools
from typing import Generator

import pytest
import torch

from .attri_util import (
    DEFAULT_BMNK_BLAS,
    DEFAULT_METRICS,
    DEFAULT_MNK_BLAS,
    FLOAT_DTYPES,
    BenchLevel,
)
from .conftest import Config
from .performance_utils import Benchmark


class BlasBenchmark(Benchmark):
    """
    benchmark for blas
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]
    DEFAULT_DTYPES = FLOAT_DTYPES
    DEFAULT_SHAPES = DEFAULT_BMNK_BLAS

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def get_input_iter(self, cur_dtype) -> Generator:
        for b, m, n, k in self.shapes:
            yield from self.input_fn(b, m, n, k, cur_dtype, self.device)

    def set_shapes(self):
        # self.shapes is a list of tuples, each containing four elements:
        # (B, M, N, K).
        self.shapes = self.DEFAULT_SHAPES[:]
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            # B=1, M=13, N=2, K=2^6..2^15
            large_k_shapes = list(
                itertools.product([1], [13], [2], [2**i for i in range(6, 15)])
            )
            # TODO: more shapes
            self.shapes.extend(large_k_shapes)

    # TODO: register_metric
    # def tflops(self):


def addmm_input_fn(b, m, n, k, cur_dtype, device):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
    bias = torch.randn([m, n], dtype=cur_dtype, device=device)
    yield inp1, inp2, bias


def bmm_input_fn(b, m, n, k, cur_dtype, device):
    inp1 = torch.randn([b, m, k], dtype=cur_dtype, device=device)
    inp2 = torch.randn([b, k, n], dtype=cur_dtype, device=device)
    yield inp1, inp2


def mm_input_fn(b, m, n, k, cur_dtype, device):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
    yield inp1, inp2


def mv_input_fn(b, m, n, k, cur_dtype, device):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    inp2 = torch.randn([k], dtype=cur_dtype, device=device)
    yield inp1, inp2


def outer_input_fn(b, m, n, k, cur_dtype, device):
    inp1 = torch.randn([m], dtype=cur_dtype, device=device)
    inp2 = torch.randn([n], dtype=cur_dtype, device=device)
    yield inp1, inp2


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "addmm",
            torch.addmm,
            addmm_input_fn,
            marks=pytest.mark.addmm(recommended_shapes=DEFAULT_MNK_BLAS),
        ),
        pytest.param(
            "bmm",
            torch.bmm,
            bmm_input_fn,
            marks=pytest.mark.bmm(recommended_shapes=DEFAULT_BMNK_BLAS),
        ),
        pytest.param(
            "mm",
            torch.Tensor.mm,
            mm_input_fn,
            marks=pytest.mark.mm,
        ),
        pytest.param(
            "mv",
            torch.Tensor.mv,
            mv_input_fn,
            marks=pytest.mark.mv,
        ),
        pytest.param(
            "outer",
            torch.Tensor.outer,
            outer_input_fn,
            marks=pytest.mark.outer,
        ),
    ],
)
def test_blas_benchmark(op_name, torch_op, input_fn):
    bench = BlasBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=FLOAT_DTYPES
    )
    bench.run()
