import pytest
import torch

from . import base, consts


class AvgPool1dBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (4, 3, 224),
            (16, 64, 128),
            (32, 128, 64),
            (64, 256, 32),
            (128, 512, 16),
        ]


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    kernel_size = [3]
    stride = [2]
    padding = [1]
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={"kernel_size": kernel_size, "stride": stride, "padding": padding},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device)
    return inp, plan.params


@pytest.mark.avg_pool1d
def test_avg_pool1d():
    AvgPool1dBenchmark(
        op_name="avg_pool1d",
        torch_op=torch.ops.aten.avg_pool1d,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
