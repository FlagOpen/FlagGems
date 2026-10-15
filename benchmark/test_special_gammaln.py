import pytest
import torch

from . import base, consts


class SpecialGammalnOutBenchmark(base.UnaryPointwiseOutBenchmark):
    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape), "out": list(shape)},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        inp = base.generate_tensor_input(shape, case.dtype, self.device)
        out = torch.empty_like(inp)
        return inp, {"out": out}


@pytest.mark.special_gammaln
def test_special_gammaln():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_gammaln",
        torch_op=torch.special.gammaln,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_gammaln_out
def test_special_gammaln_out():
    bench = SpecialGammalnOutBenchmark(
        op_name="special_gammaln_out",
        torch_op=torch.special.gammaln,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
