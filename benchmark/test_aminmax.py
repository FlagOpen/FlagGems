# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import base, consts, utils


def _aminmax_variants(shape):
    # Test dim=None (whole tensor reduction); dim=-1; and dim=0 when ndim > 1.
    variants = [{}, {"dim": -1}]
    if len(shape) > 1:
        variants.append({"dim": 0})
    return variants


def _aminmax_build(shape, kwargs, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    if kwargs:
        return inp, kwargs
    return (inp,)


class AminmaxBenchmark(base.UnaryReductionBenchmark):
    def get_case_iter(self, dtype):
        ordinal = 0
        for shape in self.shapes:
            for kwargs in _aminmax_variants(shape):
                yield self._case_from_plan(
                    dtype,
                    ordinal,
                    base.BenchmarkCasePlan(
                        shape={"input": shape},
                        params=dict(kwargs),
                        builder_args=(shape, kwargs),
                    ),
                )
                ordinal += 1

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, kwargs = plan.builder_args
        return _aminmax_build(shape, kwargs, case.dtype, self.device)


@pytest.mark.aminmax
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_aminmax():
    bench = AminmaxBenchmark(
        op_name="aminmax",
        torch_op=torch.aminmax,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


# ---------------------------------------------------------------------------
# aten::_aminmax / aten::_aminmax.out
#
# Operator ids stay `_aminmax` / `_aminmax_out`; the pytest markers are spelled
# `underscore_aminmax` / `underscore_aminmax_out` because pytest will not build a
# marker from an attribute beginning with an underscore.
# ---------------------------------------------------------------------------


class AminmaxAllReduceBenchmark(base.UnaryReductionBenchmark):
    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        inp = utils.generate_tensor_input(shape, case.dtype, self.device)
        # ``_aminmax`` reduces the whole tensor; there is no dim argument.
        return (inp,)


class AminmaxOutBenchmark(base.UnaryReductionBenchmark):
    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        cur_dtype = case.dtype
        inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
        min_out = torch.empty((), dtype=cur_dtype, device=self.device)
        max_out = torch.empty((), dtype=cur_dtype, device=self.device)
        # The aten ``.out`` overload takes keyword-only ``out0``/``out1``.
        return inp, {"out0": min_out, "out1": max_out}


@pytest.mark.underscore_aminmax
def test__aminmax():
    bench = AminmaxAllReduceBenchmark(
        op_name="_aminmax",
        torch_op=torch._aminmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.underscore_aminmax_out
def test__aminmax_out():
    bench = AminmaxOutBenchmark(
        op_name="_aminmax_out",
        torch_op=torch.ops.aten._aminmax.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
