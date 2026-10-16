import pytest
import torch

import flag_gems

from . import base


LINALG_COND_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
]


class LinalgCondBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LINALG_COND_SHAPES
        self.shape_desc = "N, N"

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={"p": None},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        n = shape[-1]
        matrix = torch.randn(shape, dtype=case.dtype, device=self.device)
        matrix = matrix + n * torch.eye(n, dtype=case.dtype, device=self.device)
        return matrix, None


@pytest.mark.linalg_cond
def test_linalg_cond():
    bench = LinalgCondBenchmark(
        op_name="linalg_cond",
        torch_op=torch.linalg.cond,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.linalg_cond)
    bench.run()
