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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark


_SHAPES = [
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 1280),
    (2, 19, 7),
]
_COMPREHENSIVE_SHAPES = [
    (1024, 1),
    (1024, 16),
    (1024, 256),
    (1024, 4096),
    (1024, 65536),
    (64, 64, 1),
    (64, 64, 16),
    (64, 64, 256),
    (64, 64, 4096),
    (64, 64, 65536),
]


class ShiftedChebyshevPolynomialTBenchmark(
    base.BinaryPointwiseBenchmark, OperatorBenchmark
):
    """Keep binary metrics while using reviewed case-listing inputs."""

    def supports_cases(self):
        return True

    def set_shapes(self, shape_file_path=None):
        OperatorBenchmark.set_shapes(self, shape_file_path, default_shapes=_SHAPES)
        if (
            base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
            and not base.Config.query
        ):
            self.shapes = list(
                dict.fromkeys([*self.shapes, *_COMPREHENSIVE_SHAPES])
            )

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs

    def get_tflops(self, op, *args, **kwargs):
        del op, kwargs
        return args[0].numel()


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for degree_kind in ("scalar", "tensor"):
        yield base.BenchmarkCasePlan(
            shape={"x": list(dims), "n": list(dims) if degree_kind == "tensor" else None},
            params={"degree_kind": degree_kind, "degree": 3},
            builder_args=(dims, degree_kind),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, degree_kind = plan.builder_args
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device=device)
    degree = (
        torch.randint(0, 10, shape, dtype=torch.int64, device=device)
        if degree_kind == "tensor"
        else 3
    )
    return x, degree


@pytest.mark.special_shifted_chebyshev_polynomial_t
def test_special_shifted_chebyshev_polynomial_t():
    bench = ShiftedChebyshevPolynomialTBenchmark(
        op_name="special_shifted_chebyshev_polynomial_t",
        torch_op=torch.special.shifted_chebyshev_polynomial_t,
        gems_op=flag_gems.special_shifted_chebyshev_polynomial_t,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    )
    bench.run()
