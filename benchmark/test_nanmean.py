import pytest
import torch

import flag_gems

from . import base, consts


class _NanmeanBenchmark(base.UnaryReductionBenchmark):
    def get_case_iter(self, dtype):
        ordinal = 0
        for shape in self.shapes:
            ndim = len(shape)
            dims = [None]
            if ndim >= 2:
                dims += [-1, 0]
            if ndim >= 3:
                dims += [1]
            for dim in dims:
                params = {} if dim is None else {"dim": dim}
                yield self._case_from_plan(
                    dtype,
                    ordinal,
                    base.BenchmarkCasePlan(
                        shape={"input": list(shape)},
                        params=params,
                        builder_args=(shape, dim),
                    ),
                )
                ordinal += 1

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, dim = plan.builder_args
        cur_dtype = case.dtype
        x = torch.randn(shape, dtype=cur_dtype, device=self.device) * 10
        mask = torch.rand(shape, device=self.device) > 0.7
        x[mask] = float("nan")
        if dim is None:
            return (x,)
        return x, dim


class _NanmeanOutBenchmark(base.UnaryReductionBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device) * 10
            x[torch.rand(shape, device=self.device) > 0.7] = float("nan")

            yield x, {"out": torch.empty((), dtype=cur_dtype, device=self.device)}

            if x.ndim >= 2:
                for dim in (-1, 0):
                    out_shape = torch.nanmean(x, dim=dim).shape
                    out = torch.empty(out_shape, dtype=cur_dtype, device=self.device)
                    yield x, dim, {"out": out}

            if x.ndim >= 3:
                out_shape = torch.nanmean(x, dim=1).shape
                out = torch.empty(out_shape, dtype=cur_dtype, device=self.device)
                yield x, 1, {"out": out}


@pytest.mark.nanmean
def test_benchmark_nanmean():
    bench = _NanmeanBenchmark(
        op_name="nanmean",
        torch_op=torch.nanmean,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.nanmean)
    bench.run()


@pytest.mark.nanmean_out
def test_benchmark_nanmean_out():
    bench = _NanmeanOutBenchmark(
        op_name="nanmean_out",
        torch_op=torch.nanmean,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.nanmean_out)
    bench.run()
