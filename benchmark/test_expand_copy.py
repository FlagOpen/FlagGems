import pytest
import torch

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    target = tuple(dims[:-1]) + (2,) if dims[-1] == 1 else dims
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims), "size": list(target)},
        params={"size": target},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return utils.generate_tensor_input(shape, dtype, device), plan.params["size"]


class ExpandCopyBenchmark(base.UnaryPointwiseBenchmark, OperatorBenchmark):
    def supports_cases(self):
        return True

    def set_shapes(self, shape_file_path=None):
        base.Benchmark.set_shapes(self, shape_file_path)

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


@pytest.mark.expand_copy
def test_expand_copy():
    ExpandCopyBenchmark(
        op_name="expand_copy",
        torch_op=torch.ops.aten.expand_copy,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
