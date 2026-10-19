import pytest
import torch

from . import base, consts


CONV_TBC_BENCH_SHAPES = [
    (16, 8, 32, 64, 3, 1),
    (64, 16, 64, 128, 5, 2),
    (128, 32, 128, 256, 3, 1),
]


class ConvTBCBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CONV_TBC_BENCH_SHAPES


def _case_fn(shape, dtype):
    del dtype
    T, B, Cin, Cout, kW, pad = shape
    yield base.BenchmarkCasePlan(
        shape={"input": [T, B, Cin], "weight": [kW, Cin, Cout], "bias": [Cout]},
        params={"pad": pad},
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    T, B, Cin, Cout, kW, pad = plan.builder_args[0]
    inp = torch.randn((T, B, Cin), device=device, dtype=dtype)
    weight = torch.randn((kW, Cin, Cout), device=device, dtype=dtype)
    bias = torch.randn((Cout,), device=device, dtype=dtype)
    return inp, weight, bias, pad


@pytest.mark.conv_tbc
def test_conv_tbc():
    ConvTBCBenchmark(
        op_name="conv_tbc",
        torch_op=torch.conv_tbc,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
