import pytest
import torch

from . import base, consts


class SpecialLegendrePolynomialPBenchmark(base.Benchmark):
    """Benchmark for special_legendre_polynomial_p (Legendre polynomial).

    This is a binary operation where the first input is a tensor and the
    second input is a scalar polynomial degree.
    """

    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={"n": 3},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        # x is the input tensor, n is the polynomial degree (scalar)
        shape = case.builder_args[0].builder_args[0]
        x = base.generate_tensor_input(shape, case.dtype, self.device)
        n = 3
        return x, n

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


@pytest.mark.special_legendre_polynomial_p
def test_special_legendre_polynomial_p():
    bench = SpecialLegendrePolynomialPBenchmark(
        op_name="special_legendre_polynomial_p",
        torch_op=torch.special.legendre_polynomial_p,
        # special.legendre_polynomial_p only supports float32 in PyTorch
        dtypes=[torch.float32],
    )
    bench.run()
