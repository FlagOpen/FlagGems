import pytest
import torch

from . import base, consts, utils


def case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        builder_args=(shape,),
    )


def build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    # dim defaults to -1 in torch.cumulative_trapezoid; return the tensor alone so
    # the benchmark records a bare shape detail.
    return (inp,)


@pytest.mark.cumulative_trapezoid
def test_cumulative_trapezoid():
    bench = base.GenericBenchmark2DOnly(
        case_fn=case_fn,
        build_inputs_fn=build_inputs_fn,
        op_name="cumulative_trapezoid",
        torch_op=torch.cumulative_trapezoid,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
