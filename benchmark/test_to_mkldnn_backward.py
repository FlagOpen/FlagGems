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

from functools import partial

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# Both native-valid call forms are timed for every shape: matching dtypes (the
# operator hands `grad` back untouched) and a dtype cast (a fresh tensor cast to
# input.dtype).  Only dtypes are read, so the payloads stay uninitialized and
# are never compared.  Extra shapes are unioned with the shared shape set, which
# keeps core_shapes.yaml, DEFAULT_SHAPES and a caller --shape-file in effect.
EXTRA_SHAPES = [(1024, 1024), (20, 320, 15), (16, 128, 64)]

DENSE_GROUP = "dense"
OPAQUE_GROUP = "opaque"
OPAQUE_CROSS_CAST_GROUP = "opaque-cross-cast"
OPAQUE_MODE = "opaque to dense"
_CROSS_CAST_DTYPES = {
    (str(torch.float16), str(torch.bfloat16)),
    (str(torch.bfloat16), str(torch.float16)),
}

_DEVICE_DTYPE_FLAG = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
    torch.int64: "support_int64",
}


def _dtype_supported(dtype):
    flag = _DEVICE_DTYPE_FLAG.get(dtype)
    return True if flag is None else bool(getattr(flag_gems.runtime.device, flag))


BENCH_DTYPES = [
    dtype
    for group in (
        consts.FLOAT_DTYPES,
        consts.INT_DTYPES,
        consts.EXTRA_INT_DTYPES,
        consts.BOOL_DTYPES,
        [
            torch.float64,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.complex64,
            torch.complex128,
        ],
    )
    for dtype in group
    if _dtype_supported(dtype)
]

_CAST_PARTNER = {
    torch.float64: torch.float32,
    torch.float8_e4m3fn: torch.float32,
    torch.float8_e5m2: torch.float32,
    torch.float16: torch.float32,
    torch.float32: torch.float16,
    torch.bfloat16: torch.float32,
    torch.int16: torch.int32,
    torch.int32: torch.int16,
    torch.int8: torch.int32,
    torch.uint8: torch.int32,
    torch.int64: torch.int32,
    torch.bool: torch.int32,
    torch.complex64: torch.complex128,
    torch.complex128: torch.complex64,
}


def _cast_partner(dtype):
    partner = _CAST_PARTNER.get(dtype)
    return dtype if partner is None or not _dtype_supported(partner) else partner


def _all_case_fn(shape, dtype):
    partner = _cast_partner(dtype)
    forms = [("same dtype", dtype)]
    if partner != dtype:
        forms.append(("dtype cast", partner))
    for mode, input_dtype in forms:
        yield base.BenchmarkCasePlan(
            shape={"grad": list(shape), "input": list(shape)},
            params={
                "mode": mode,
                "grad_dtype": str(dtype),
                "input_dtype": str(input_dtype),
            },
            builder_args=(shape, input_dtype),
        )

    # The actual layout-backward path consumes opaque CPU gradients. Its dtype
    # conversions differ from dense casts, and rank-zero opaque tensors do not exist.
    if shape and dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int8,
        torch.uint8,
    ):
        targets = (dtype,)
        if dtype in (torch.float32, torch.float16, torch.bfloat16):
            targets = (
                torch.float32,
                torch.float16,
                torch.bfloat16,
                torch.int8,
                torch.uint8,
            )
        for target in targets:
            yield base.BenchmarkCasePlan(
                shape={"grad": list(shape), "input": list(shape)},
                params={
                    "mode": OPAQUE_MODE,
                    "grad_dtype": str(dtype),
                    "input_dtype": str(target),
                },
                builder_args=(shape, target, "opaque"),
            )


def _group_for_plan(plan):
    if plan.params["mode"] != OPAQUE_MODE:
        return DENSE_GROUP
    dtype_pair = (plan.params["grad_dtype"], plan.params["input_dtype"])
    return OPAQUE_CROSS_CAST_GROUP if dtype_pair in _CROSS_CAST_DTYPES else OPAQUE_GROUP


def _case_fn(shape, dtype, case_group=None):
    for plan in _all_case_fn(shape, dtype):
        if case_group is None or _group_for_plan(plan) == case_group:
            yield plan


def _build_inputs_fn(plan, dtype, device):
    # The cast reads grad values; the benchmark does not compare its result,
    # so uninitialized payloads are enough (generate_tensor_input has no
    # generator for fp8/complex anyway).
    if len(plan.builder_args) == 3:
        shape, input_dtype, _ = plan.builder_args
        grad = torch.empty(shape, dtype=dtype, device="cpu").to_mkldnn()
        inp = torch.empty(shape, dtype=input_dtype, device="cpu")
        return grad, inp, {}
    shape, input_dtype = plan.builder_args
    grad = torch.empty(shape, dtype=dtype, device=device)
    inp = torch.empty(shape, dtype=input_dtype, device=device)
    return grad, inp, {}


class ToMkldnnBackwardBenchmark(OperatorBenchmark):

    # Keep the full operator dtype set for validation; execute only this node.
    partition_dtype = None

    def set_dtypes(self, user_desired_dtypes):
        super().set_dtypes(user_desired_dtypes)
        if self.partition_dtype is not None:
            if self.partition_dtype not in self.dtypes:
                raise ValueError("partition_dtype must belong to the operator's dtypes")
            if self.partition_dtype not in self.to_bench_dtypes:
                pytest.skip(f"Dtype {self.partition_dtype} was not requested")
            self.to_bench_dtypes = [self.partition_dtype]

    case_group = None

    def set_shapes(self, shape_file_path=None, *, default_shapes=None):
        super().set_shapes(shape_file_path, default_shapes=default_shapes)
        merged = [tuple(shape) for shape in self.shapes] + list(EXTRA_SHAPES)
        self.shapes = list(dict.fromkeys(merged))
        if self.case_group is not None:
            # Query mode calls set_shapes before user dtype selection.
            dtype = self.partition_dtype or self.to_bench_dtypes[0]
            self.shapes = [
                shape
                for shape in self.shapes
                if any(_case_fn(shape, dtype, self.case_group))
            ]
            if not self.shapes:
                pytest.skip("No MKLDNN backward workloads in this layout/cast group")


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize(
    "dtype,case_group",
    [(dtype, DENSE_GROUP) for dtype in BENCH_DTYPES]
    + [
        (dtype, OPAQUE_GROUP)
        for dtype in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.uint8,
        )
        if dtype in BENCH_DTYPES
    ]
    + [
        (dtype, OPAQUE_CROSS_CAST_GROUP)
        for dtype in (torch.float16, torch.bfloat16)
        if dtype in BENCH_DTYPES
    ],
    ids=str,
)
def test_to_mkldnn_backward(dtype, case_group):
    if flag_gems.vendor_name == "ascend" and case_group == OPAQUE_CROSS_CAST_GROUP:
        pytest.skip(
            "Ascend host MKLDNN cannot convert opaque gradients between FP16 and BF16"
        )
    bench = ToMkldnnBackwardBenchmark(
        op_name="to_mkldnn_backward",
        torch_op=torch.ops.aten.to_mkldnn_backward,
        case_fn=partial(_case_fn, case_group=case_group),
        build_inputs_fn=_build_inputs_fn,
        gems_op=getattr(flag_gems, "to_mkldnn_backward", None),
        dtypes=BENCH_DTYPES,
        partition_dtype=dtype,
        case_group=case_group,
    )
    bench.run()
