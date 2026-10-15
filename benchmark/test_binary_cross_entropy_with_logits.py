import pytest
import torch

from . import base, consts


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape), "target": list(shape)},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = torch.rand(shape, dtype=dtype, device=device)
    return inp, target


@pytest.mark.binary_cross_entropy_with_logits
def test_binary_cross_entropy_with_logits():
    bench = base.GenericBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="binary_cross_entropy_with_logits",
        torch_op=torch.ops.aten.binary_cross_entropy_with_logits,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
