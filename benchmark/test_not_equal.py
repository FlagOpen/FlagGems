import pytest
import torch

from . import base, consts, utils


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(shape={"self": list(dims), "other": list(dims)}, params={}, builder_args=(dims, "tensor"))


def _scalar_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(shape={"self": list(dims)}, params={"other": 0.5}, builder_args=(dims, "scalar"))


def _build_inputs_fn(plan, dtype, device):
    shape, kind = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    if kind == "scalar":
        return inp, plan.params["other"]
    return inp, utils.generate_tensor_input(shape, dtype, device)


@pytest.mark.not_equal
def test_not_equal():
    base.GenericBenchmark(op_name="not_equal", torch_op=torch.not_equal, case_fn=_case_fn, build_inputs_fn=_build_inputs_fn, dtypes=consts.FLOAT_DTYPES).run()


@pytest.mark.not_equal_scalar
def test_not_equal_scalar():
    base.GenericBenchmark(op_name="not_equal_scalar", torch_op=torch.not_equal, case_fn=_scalar_case_fn, build_inputs_fn=_build_inputs_fn, dtypes=consts.FLOAT_DTYPES).run()
