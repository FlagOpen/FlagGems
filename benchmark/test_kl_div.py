import pytest
import torch

from . import base, consts


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(shape={"input": list(dims), "target": list(dims)}, params={}, builder_args=(dims,))
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield base.BenchmarkCasePlan(shape={"input": list(dims), "target": list(dims)}, params={"log_target": True}, builder_args=(dims,))


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = torch.rand(shape, dtype=dtype, device=device)
    return (inp, target, plan.params) if plan.params else (inp, target)


@pytest.mark.kl_div
def test_kl_div():
    base.GenericBenchmark2DOnly(
        op_name="kl_div",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.kl_div,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
