import pytest
import torch

import flag_gems

from . import base, consts


CORE_SHAPES = [
    (2, 2),
    (3, 2),
    (2, 3),
    (4, 4),
    (8, 4),
    (4, 8),
    (256, 2),
    (2, 256),
    (32, 4, 4),
]
COMPREHENSIVE_SHAPES = [
    (1024, 2),
    (2, 1024),
    (256, 4),
    (4, 256),
    (128, 4, 4),
]


class PinverseBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += self.set_more_shapes()
        self.shape_desc = "batch..., M, N"

    def set_more_shapes(self):
        return list(COMPREHENSIVE_SHAPES)

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={"rcond": 1e-15},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        matrix = torch.randn(shape, dtype=case.dtype, device=self.device)
        return matrix, {"rcond": 1e-15}


@pytest.mark.pinverse
def test_pinverse():
    bench = PinverseBenchmark(
        op_name="pinverse",
        torch_op=torch.pinverse,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.pinverse)
    bench.run()
