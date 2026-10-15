import pytest
import torch

from . import base

# PDIST requires the input dim to be reasonably small; these shapes follow the upstream test suite.
PDIST_SHAPES = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]


class PdistBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PDIST_SHAPES

    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    params={"p": 2.0},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        x = torch.randn(shape, dtype=case.dtype, device=self.device)
        return (x, 2.0)


@pytest.mark.pdist
def test_pdist():
    bench = PdistBenchmark(
        op_name="pdist",
        torch_op=torch.pdist,
        # pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
        dtypes=[torch.float32],
    )
    bench.run()
