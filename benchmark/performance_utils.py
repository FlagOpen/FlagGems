import time
from typing import List, Optional

import torch
import triton

import flag_gems

from .attri_util import (
    DEFAULT_METRICS,
    DEFAULT_NON_BLAS_BENCH_SHAPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    BenchmarkMetrics,
    BenchmarkResult,
)
from .conftest import Config

torch.backends.cuda.matmul.allow_tf32 = False


class Benchmark:
    DEFAULT_METRICS = DEFAULT_METRICS
    DEFAULT_DTYPES = FLOAT_DTYPES
    DEFAULT_SHAPES = DEFAULT_NON_BLAS_BENCH_SHAPES
    """
    the base class for the operations benchmark
    """

    def __init__(
        self,
        op_name,
        torch_op,
        arg_func,
        dtypes,
        batch,
        sizes,
        is_backward=False,
        kwargs_func=None,
    ):
        self.op_name = op_name
        if is_backward:
            self.op_name += " backward"
        self.torch_op = torch_op
        self.arg_func = arg_func
        self.kwargs_func = kwargs_func
        self.dtypes = dtypes
        self.batch = batch
        self.sizes = sizes
        self.gems_op = None
        self.is_backward = is_backward
        self.results = []

    def set_metrics(self, user_desired_metrics: Optional[List[str,]]):
        self.to_bench_metrics = (
            user_desired_metrics if user_desired_metrics else self.DEFAULT_METRICS
        )

    def set_dtypes(self, user_desired_dtypes: Optional[List[torch.dtype]]):
        self.to_bench_dtypes = (
            user_desired_dtypes if user_desired_dtypes else self.DEFAULT_DTYPES
        )

    def set_gems(self, gems_op):
        self.gems_op = gems_op

    def get_latency(self, op, *args, **kwargs):
        fn = lambda: op(*args, **kwargs)
        if self.is_backward:
            out = fn()
            dout = torch.randn_like(out)
            fn = lambda: out.backward(dout, retain_graph=True)
        if Config.cpu_mode:
            for i in range(Config.warm_up):
                fn()
            torch.cuda.synchronize()
            start = time.time()
            for i in range(Config.repetition):
                fn()
            torch.cuda.synchronize()
            end = time.time()
            latency = (end - start) / Config.repetition * 1000
        else:
            latency = triton.testing.do_bench(
                fn,
                warmup=Config.warm_up,
                rep=Config.repetition,
                return_mode="median",
            )
        # average latency in ms
        return latency

    def get_tflops(self, op, *args, **kwargs):
        """This method is currently not really implemented and serves as a placeholder.
        A proper implementation will be developed in the future."""

        from torch.utils.flop_counter import FlopCounterMode

        fn = lambda: op(*args, **kwargs)
        with FlopCounterMode(display=False) as flop_counter:
            fn()
        return flop_counter.get_total_flops()

    def run(self):
        for dtype in self.dtypes:
            metrics = []
            for size in self.sizes:
                args = ()
                if self.arg_func is not None:
                    args = self.arg_func(dtype, self.batch, size)
                if self.is_backward:
                    args = tuple(
                        a.clone().requires_grad_()
                        if torch.is_tensor(a) and torch.is_floating_point(a)
                        else a
                        for a in args
                    )

                kwargs = {}
                if self.kwargs_func is not None:
                    kwargs = self.kwargs_func(dtype, self.batch, size)

                torch_latency = self.get_latency(self.torch_op, *args, **kwargs)
                if self.gems_op:
                    gems_latency = self.get_latency(self.gems_op, *args, **kwargs)
                else:
                    with flag_gems.use_gems():
                        gems_latency = self.get_latency(self.torch_op, *args, **kwargs)
                speedup = torch_latency / gems_latency

                tflops = self.get_tflops(self.torch_op, *args, **kwargs)
                utilization = tflops / gems_latency / 1e12 * 1e3

                metric = BenchmarkMetrics(
                    shape_detail=size,
                    latency_base=torch_latency,
                    latency=gems_latency,
                    speedup=speedup,
                    tflops=tflops,
                    utilization=utilization,
                )
                metrics.append(metric)

            result = BenchmarkResult(
                op_name=self.op_name,
                dtype=str(dtype),
                mode="cpu" if Config.cpu_mode else "cuda",
                result=metrics,
            )
            print(result)

def unary_arg(dtype, batch, shape):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cuda")
    elif dtype in INT_DTYPES:
        inp = torch.randint(low=0, high=0x7FFF, size=shape, dtype=dtype, device="cuda")
    return (inp,)


def binary_args(dtype, batch, shape):
    if dtype in FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device="cuda")
        inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    elif dtype in INT_DTYPES:
        inp1 = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cuda",
        )
        inp2 = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cuda",
        )
    return inp1, inp2


def ternary_args(dtype, batch, shape):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    inp3 = torch.randn(shape, dtype=dtype, device="cuda")
    return inp1, inp2, inp3
