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

import math

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# aten::view_as_complex_copy needs rank >= 1, a last dimension of size 2 with
# stride 1 and an even element storage offset, which no shape in the shared
# default shape set satisfies, so the suite declares its own scales below. Each
# timed call copies one real pair into one complex element.
VIEW_AS_COMPLEX_COPY_SHAPES = [
    (1 << 20, 2),
    (1024, 1024, 2),
    (2048, 2048, 2),
    (20, 320, 15, 2),
    (16, 128, 64, 60, 2),
    (16, 7, 57, 32, 29, 2),
]

# Exactly the dtypes aten::view_as_complex_copy accepts: float16 and float32
# everywhere, float64 only where the device reports FP64 support.
VIEW_AS_COMPLEX_COPY_DTYPES = [torch.float16, torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)


def _is_buildable(shape):
    """Native input contract for one requested shape.

    Rank 0, a last extent other than 2, and negative, bool or non-integer extents
    are not valid inputs; valid empty leading axes such as (0, 2) and a scalar
    output shape (2,) are kept.
    """
    return (
        isinstance(shape, (tuple, list))
        and len(shape) >= 1
        and shape[-1] == 2
        and all(
            isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0
            for dim in shape
        )
    )


def _case_fn(shape, dtype):
    del dtype
    # A caller-supplied shape file may list shapes this operator cannot accept;
    # report them instead of silently dropping requested work.
    if not _is_buildable(shape):
        raise ValueError(
            "view_as_complex_copy needs rank >= 1, a last dimension of size 2 and "
            f"non-negative integer extents, but got {shape!r}"
        )
    yield base.BenchmarkCasePlan(shape={"input": list(shape)}, builder_args=(shape,))


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    # One fresh contiguous real tensor, which is this operator's only argument.
    return torch.randn(shape, dtype=dtype, device=device), {}


class ViewAsComplexCopyBenchmark(OperatorBenchmark):
    # Keep the full operator dtype set for validation; execute only this node.
    partition_dtype = None
    empty_input = None

    def set_dtypes(self, user_desired_dtypes):
        super().set_dtypes(user_desired_dtypes)
        if self.partition_dtype is None:
            return
        if self.partition_dtype not in self.dtypes:
            raise ValueError("partition_dtype must belong to the operator's dtypes")
        if self.partition_dtype not in self.to_bench_dtypes:
            pytest.skip(f"Dtype {self.partition_dtype} was not requested")
        self.to_bench_dtypes = [self.partition_dtype]

    def set_shapes(self, shape_file_path=None):
        # A caller-supplied shape file still wins; these scales are the fallback
        # because no shared default shape is a valid input for this operator.
        super().set_shapes(shape_file_path, default_shapes=VIEW_AS_COMPLEX_COPY_SHAPES)
        if self.empty_input is not None:
            self.shapes = [
                shape
                for shape in self.shapes
                if (math.prod(shape) == 0) == self.empty_input
            ]
            if not self.shapes:
                pytest.skip("No workloads in this input-emptiness group")


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("dtype", VIEW_AS_COMPLEX_COPY_DTYPES, ids=str)
@pytest.mark.parametrize(
    "empty_input", [False, True], ids=["nonempty-input", "empty-input"]
)
def test_view_as_complex_copy(dtype, empty_input):
    if flag_gems.vendor_name == "ascend" and dtype == torch.float16 and not empty_input:
        pytest.skip("Ascend copy does not support complex32 output")

    bench = ViewAsComplexCopyBenchmark(
        op_name="view_as_complex_copy",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.view_as_complex_copy,
        gems_op=getattr(flag_gems, "view_as_complex_copy", None),
        dtypes=VIEW_AS_COMPLEX_COPY_DTYPES,
        partition_dtype=dtype,
        empty_input=empty_input,
    )
    bench.run()
