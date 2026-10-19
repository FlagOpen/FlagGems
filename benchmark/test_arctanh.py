# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)}, params={}, builder_args=(dims,)
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return (utils.generate_tensor_input(shape, dtype, device),)


class ArctanhBenchmark(base.UnaryPointwiseBenchmark, OperatorBenchmark):
    def supports_cases(self):
        return True

    def set_shapes(self, shape_file_path=None):
        base.Benchmark.set_shapes(self, shape_file_path)

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


@pytest.mark.arctanh
@pytest.mark.skipif(flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working")
def test_arctanh():
    ArctanhBenchmark(
        op_name="arctanh",
        torch_op=torch.arctanh,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()


@pytest.mark.arctanh_
@pytest.mark.skipif(flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working")
def test_arctanh_inplace():
    ArctanhBenchmark(
        op_name="arctanh_",
        torch_op=lambda a: a.arctanh_(),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    ).run()
