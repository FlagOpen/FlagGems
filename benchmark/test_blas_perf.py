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
    llama_shapes,
)
from .conftest import Config
from .performance_utils import Benchmark

MV_RECOMMENDED_SHAPES = [(m, n) for m, n, k in DEFAULT_MNK_BLAS]

OUTER_RECOMENDED_SHAPES = [(m, n) for m, n, k in DEFAULT_MNK_BLAS[:-1]] + [
    (10240, 10240)  # from perf
]


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
        # llama shapes
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            for m, n, k in llama_shapes():
                yield from self.input_fn(1, m, n, k, cur_dtype, self.device)

    def set_shapes(self):
        # self.shapes is a list of tuples, each containing four elements:
        # (B, M, N, K).
        self.shapes = self.DEFAULT_SHAPES[:]
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            # 'mv' operations only involve M and N dimensions.
            # Shapes with large K values are not suitable for these two operations.
            if self.op_name not in ["mv"]:
                # B=1 or 4, M= 13, N= 2 , K=2^6..2^15
                large_k_shapes = list(
                    itertools.product([1, 4], [13], [2], [2**i for i in range(6, 15)])
                )
                self.shapes.extend(large_k_shapes)

            split_k_shapes = [
                (1, m, m, k)
                for m in [16 * i for i in range(1, 5)]
                for k in [4096 * i for i in range(1, 9)]
            ]
            self.shapes.extend(split_k_shapes)

    def get_tflops(self, op, *args, **kwargs):
        """This method is currently not really implemented and serves as a placeholder.
        A proper implementation will be developed in the future."""
        from torch.utils.flop_counter import FlopCounterMode

        fn = lambda: op(*args, **kwargs)
        with FlopCounterMode(display=False) as flop_counter:
            fn()
        tflops = flop_counter.get_total_flops()
        return tflops


def addmm_input_fn(b, m, n, k, cur_dtype, device):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
    bias = torch.randn([m, n], dtype=cur_dtype, device=device)
    yield bias, inp1, inp2,


def bmm_input_fn(b, m, n, k, cur_dtype, device):
    inp1 = torch.randn([b, m, k], dtype=cur_dtype, device=device)
    inp2 = torch.randn([b, k, n], dtype=cur_dtype, device=device)
    yield inp1, inp2


def mm_input_fn(b, m, n, k, cur_dtype, device):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
    yield inp1, inp2


def mv_input_fn(b, m, n, k, cur_dtype, device):
    inp1 = torch.randn([m, n], dtype=cur_dtype, device=device)
    inp2 = torch.randn([n], dtype=cur_dtype, device=device)
    yield inp1, inp2


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "addmm",
            torch.addmm,
            addmm_input_fn,
            marks=pytest.mark.addmm(
                recommended_shapes=DEFAULT_MNK_BLAS, shape_desc="M, N, K"
            ),
        ),
        pytest.param(
            "bmm",
            torch.bmm,
            bmm_input_fn,
            marks=pytest.mark.bmm(
                recommended_shapes=DEFAULT_BMNK_BLAS, shape_desc="B, M, N, K"
            ),
        ),
        pytest.param(
            "mm",
            torch.Tensor.mm,
            mm_input_fn,
            marks=pytest.mark.mm(
                recommended_shapes=DEFAULT_MNK_BLAS, shape_desc="M, N, K"
            ),
        ),
        pytest.param(
            "mv",
            torch.Tensor.mv,
            mv_input_fn,
            marks=pytest.mark.mv(recommended_shapes=MV_RECOMMENDED_SHAPES),
        ),
    ],
)
def test_blas_benchmark(op_name, torch_op, input_fn):
    bench = BlasBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=FLOAT_DTYPES
    )
    bench.run()


class OuterBenchmark(BlasBenchmark):
    """
    benchmark for outer
    """

    DEFAULT_SHAPES = OUTER_RECOMENDED_SHAPES

    def set_shapes(self):
        # 'outer' only involve M and N dimensions.
        self.shapes = self.DEFAULT_SHAPES[:]

    def get_input_iter(self, cur_dtype) -> Generator:
        for m, n in self.shapes:
            yield from self.input_fn(m, n, cur_dtype, self.device)


@pytest.mark.outer(recommended_shapes=OUTER_RECOMENDED_SHAPES)
def test_outer_benchmark():
    def outer_input_fn(m, n, cur_dtype, device):
        inp1 = torch.randn([m], dtype=cur_dtype, device=device)
        inp2 = torch.randn([n], dtype=cur_dtype, device=device)
        yield inp1, inp2

    bench = OuterBenchmark(
        input_fn=outer_input_fn,
        op_name="outer",
        torch_op=torch.Tensor.outer,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
