import pytest
import torch

from . import base, consts, utils


def _case_fn_scalar(shape, cur_dtype):
    yield base.BenchmarkCasePlan(
        shape={"inp": list(shape)},
        builder_args=(shape,),
    )


def _build_inputs_fn_scalar(plan, cur_dtype, device):
    (shape,) = plan.builder_args
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    return inp, 0


@pytest.mark.less_equal_
def test_less_equal_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="less_equal_",
        torch_op=lambda a, b: torch.ops.aten.less_equal_.Tensor(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.less_equal_scalar_
def test_less_equal_scalar_():
    bench = base.GenericBenchmark(
        op_name="less_equal_scalar_",
        case_fn=_case_fn_scalar,
        build_inputs_fn=_build_inputs_fn_scalar,
        torch_op=lambda a, b: torch.ops.aten.less_equal_.Scalar(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
