import math

import pytest
import torch

from . import base, consts
from .generated_operator_utils import OperatorBenchmark


PUT_BENCH_SHAPES = [(16,), (64, 64), (20, 320, 15), (16, 128, 64, 60)]
PUT_MAX_NUMEL = 2**28


class PutBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        base.Benchmark.set_shapes(self, shape_file_path)
        self.shapes = [s for s in self.shapes if math.prod(s) < PUT_MAX_NUMEL]

    def set_more_shapes(self):
        return PUT_BENCH_SHAPES


def _case_fn_for(accumulate):
    def case_fn(shape, dtype):
        del dtype
        dims = tuple(shape)
        count = max(1, min(math.prod(dims) // 2, 2**20))
        yield base.BenchmarkCasePlan(shape={"self": list(dims), "index": [count], "source": [count]}, params={"accumulate": accumulate}, builder_args=(dims, count))
    return case_fn


def _build_inputs_fn(plan, dtype, device):
    shape, count = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device)
    numel = inp.numel()
    if plan.params["accumulate"]:
        index = torch.randint(0, numel, (count,), dtype=torch.int64, device=device)
    else:
        index = torch.randperm(numel, device=device)[:count].to(torch.int64)
    source = torch.randn(count, dtype=dtype, device=device)
    return inp, index, source, plan.params["accumulate"]


@pytest.mark.put_
def test_put_():
    PutBenchmark(op_name="put_", torch_op=torch.ops.aten.put_, case_fn=_case_fn_for(False), build_inputs_fn=_build_inputs_fn, dtypes=consts.FLOAT_DTYPES, is_inplace=True, fresh_inputs=True).run()


@pytest.mark.put_
def test_put__accumulate():
    PutBenchmark(op_name="put_", torch_op=torch.ops.aten.put_, case_fn=_case_fn_for(True), build_inputs_fn=_build_inputs_fn, dtypes=consts.FLOAT_DTYPES, is_inplace=True, fresh_inputs=True).run()
