import pytest
import torch

from . import base

# linalg_vander benchmark
VANDER_SHAPES = [
    (4,),
    (8,),
    (16,),
    (32,),
    (64,),
    (128,),
    (256,),
    (512,),
    (1024,),
    (2, 4),
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
]


class LinalgVanderBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = VANDER_SHAPES

    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"x": list(shape)},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        x = torch.randn(shape, dtype=case.dtype, device=self.device)
        return x, {}


@pytest.mark.linalg_vander
def test_linalg_vander():
    bench = LinalgVanderBenchmark(
        op_name="linalg_vander",
        torch_op=torch.linalg.vander,
        # torch.linalg.vander only supports float32 on GPU;
        # float16/bf16 are not accelerated and produce low-precision results
        dtypes=[torch.float32],
    )
    bench.run()
