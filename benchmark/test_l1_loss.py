import pytest
import torch

import flag_gems

from . import base, consts


class L1LossBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPES = [(262144,), (1024, 1024), (4096, 4096), (64, 512, 512)]
    DEFAULT_SHAPE_DESC = "(B), M, N"

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            dims = tuple(shape)
            yield self._case_from_plan(dtype, ordinal, base.BenchmarkCasePlan(
                shape={"input": list(dims), "target": list(dims)},
                params={"reduction": 1},
                builder_args=(dims,),
            ))

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        return (
            torch.randn(shape, device=self.device, dtype=case.dtype),
            torch.randn(shape, device=self.device, dtype=case.dtype),
            plan.params["reduction"],
        )


@pytest.mark.l1_loss
@pytest.mark.skipif(flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working")
def test_l1_loss():
    L1LossBenchmark(op_name="l1_loss", torch_op=torch.ops.aten.l1_loss, dtypes=consts.FLOAT_DTYPES).run()
