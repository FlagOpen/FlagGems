import pytest
import torch

from . import base


CHOLESKY_INVERSE_SHAPES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
]


class CholeskyInverseBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_INVERSE_SHAPES
        self.shape_desc = "N, N"

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"factor": list(shape)},
                    params={"upper": False},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        n = shape[-1]
        matrix = torch.randn(shape, dtype=case.dtype, device=self.device)
        positive_definite = (
            matrix @ matrix.transpose(-2, -1)
            + torch.eye(n, dtype=case.dtype, device=self.device) * n
        )
        factor = torch.linalg.cholesky(positive_definite)
        return factor, {"upper": False}


@pytest.mark.cholesky_inverse
def test_cholesky_inverse():
    CholeskyInverseBenchmark(
        op_name="cholesky_inverse",
        torch_op=torch.cholesky_inverse,
        dtypes=[torch.float32, torch.float64],
    ).run()
