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


_POINT_SHAPES = [(1024, 2), (4096, 2), (16384, 2), (4096, 3), (8192, 4)]
_CPU_CACHE = {}


def _stage(tensor):
    key = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
    _CPU_CACHE[key] = tensor.detach().to("cpu")


def _cpu(tensor):
    key = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
    staged = _CPU_CACHE.get(key)
    return tensor.detach().to("cpu") if staged is None else staged


class HistogramddFromBinTensorsBenchmark(OperatorBenchmark):
    DEFAULT_METRICS = ["latency_base", "latency"]

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_POINT_SHAPES)


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for mode in ("plain", "weighted", "density"):
        yield base.BenchmarkCasePlan(
            shape={"input": list(dims)},
            params={"mode": mode, "edges": 11},
            builder_args=(dims, mode),
        )


def _out_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims), "out": [10] * dims[-1]},
        params={"mode": "out", "edges": 11},
        builder_args=(dims,),
    )


def _make_inputs(shape, dtype, device):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    bins = tuple(
        torch.linspace(-3.0, 3.0, 11, dtype=dtype, device=device)
        for _ in range(shape[-1])
    )
    _stage(inp)
    for edge in bins:
        _stage(edge)
    return inp, bins


def _build_inputs_fn(plan, dtype, device):
    shape, mode = plan.builder_args
    inp, bins = _make_inputs(shape, dtype, device)
    kwargs = {}
    if mode == "weighted":
        weight = torch.rand(shape[:-1], dtype=dtype, device=device)
        _stage(weight)
        kwargs["weight"] = weight
    elif mode == "density":
        kwargs["density"] = True
    return inp, bins, kwargs


def _out_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp, bins = _make_inputs(shape, dtype, device)
    out = torch.empty((10,) * shape[-1], dtype=dtype, device=device)
    return inp, bins, {"out": out}


def _cpu_reference(inp, bins, *, weight=None, density=False):
    return torch._histogramdd_from_bin_tensors(
        _cpu(inp),
        tuple(_cpu(edge) for edge in bins),
        weight=None if weight is None else _cpu(weight),
        density=density,
    )


def _cpu_out_reference(
    inp, bins, *, weight=None, density=False, out=None
):
    del out
    cpu_out = torch.empty(
        tuple(edge.numel() - 1 for edge in bins), dtype=inp.dtype, device="cpu"
    )
    torch.ops.aten._histogramdd_from_bin_tensors.out(
        _cpu(inp),
        tuple(_cpu(edge) for edge in bins),
        weight=None if weight is None else _cpu(weight),
        density=density,
        out=cpu_out,
    )
    return cpu_out


@pytest.mark.histogramdd_from_bin_tensors
def test_histogramdd_from_bin_tensors():
    bench = HistogramddFromBinTensorsBenchmark(
        op_name="histogramdd_from_bin_tensors",
        torch_op=_cpu_reference,
        gems_op=flag_gems._histogramdd_from_bin_tensors,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.histogramdd_from_bin_tensors
def test_histogramdd_from_bin_tensors_out():
    bench = HistogramddFromBinTensorsBenchmark(
        op_name="histogramdd_from_bin_tensors_out",
        torch_op=_cpu_out_reference,
        gems_op=flag_gems._histogramdd_from_bin_tensors_out,
        case_fn=_out_case_fn,
        build_inputs_fn=_out_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
