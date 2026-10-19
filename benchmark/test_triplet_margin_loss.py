import pytest
import torch

from . import base, consts


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(shape={"anchor": list(dims), "positive": list(dims), "negative": list(dims)}, params={}, builder_args=(dims,))


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return tuple(torch.randn(shape, dtype=dtype, device=device) for _ in range(3))


@pytest.mark.triplet_margin_loss
def test_triplet_margin_loss():
    base.GenericBenchmark(op_name="triplet_margin_loss", case_fn=_case_fn, build_inputs_fn=_build_inputs_fn, torch_op=torch.ops.aten.triplet_margin_loss, dtypes=consts.FLOAT_DTYPES).run()
