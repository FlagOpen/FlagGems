import pytest
import torch

from . import base, consts

# Shapes for diagonal_scatter benchmark
DIAGONAL_SCATTER_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (64, 128),
    (128, 256),
    (256, 512),
    (32, 64, 64),
    (64, 128, 128),
    (16, 32, 32, 32),
]


class DiagonalScatterBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = DIAGONAL_SCATTER_SHAPES

    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={"offset": 0, "dim1": -2, "dim2": -1},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        inp = torch.randn(shape, dtype=case.dtype, device=self.device)
        # Get diagonal to size src tensor correctly
        diag = torch.diagonal(inp, 0, -2, -1)
        src = torch.randn(diag.shape, dtype=case.dtype, device=self.device)
        return inp, src, 0, -2, -1


@pytest.mark.diagonal_scatter
def test_diagonal_scatter():
    bench = DiagonalScatterBenchmark(
        op_name="diagonal_scatter",
        torch_op=torch.diagonal_scatter,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
