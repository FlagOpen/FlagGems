import time
from typing import Any, Generator, List, Optional, Tuple

import torch
import triton

import flag_gems

from .attri_util import (
    DEFAULT_METRICS,
    DEFAULT_NON_BLAS_BENCH_SHAPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    BenchLevel,
    BenchmarkMetrics,
    BenchmarkResult,
)
from .conftest import Config

torch.backends.cuda.matmul.allow_tf32 = False


class Benchmark:
    device: str = "cuda"
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
        arg_func=None,
        dtypes=None,
        batch=1,
        sizes=None,
        is_backward=False,
        kwargs_func=None,
    ):
        self.op_name = op_name
        if is_backward:
            self.op_name += " backward"
        self.torch_op = torch_op
        self.gems_op = None
        self.is_backward = is_backward
        self._input_iter = None

        # Theoretical supported dtypes, metrics, for the operation.
        # These are set by default.
        self.dtypes = dtypes if dtypes is not None else self.DEFAULT_DTYPES
        self.metrics = self.DEFAULT_METRICS

        # Actual dtypes and metrics to be used in the benchmark,
        # can be influenced by user input.
        self.to_bench_dtypes = self.dtypes
        self.to_bench_metrics = self.metrics

        self.shapes = sizes if sizes is not None else self.DEFAULT_SHAPES

        self.batch = batch
        self.arg_func = arg_func
        self.kwargs_func = kwargs_func

    def set_metrics(self, user_desired_metrics: Optional[List[str]]):
        # Validate user-specified metrics
        if user_desired_metrics and not all(
            metric in self.metrics for metric in user_desired_metrics
        ):
            invalid_metrics = [
                metric for metric in user_desired_metrics if metric not in self.metrics
            ]
            raise ValueError(
                f"Given metric(s) '{', '.join(invalid_metrics)}' can't be supported by this op '{self.op_name}'"
            )

        self.to_bench_metrics = (
            user_desired_metrics if user_desired_metrics else self.metrics
        )

    def set_dtypes(self, user_desired_dtypes: Optional[List[torch.dtype]]):
        # Validate user-specified dtypes
        if user_desired_dtypes and not all(
            dtype in self.dtypes for dtype in user_desired_dtypes
        ):
            invalid_dtypes = [
                dtype for dtype in user_desired_dtypes if dtype not in self.dtypes
            ]
            raise ValueError(
                f"Given dtype(s) '{', '.join(str(dtype) for dtype in invalid_dtypes)}'"
                f"can't be supported by this op '{self.op_name}'"
            )
        self.to_bench_dtypes = (
            user_desired_dtypes if user_desired_dtypes else self.dtypes
        )

    def set_shapes(self):
        self.shapes = self.DEFAULT_SHAPES

    def record_shapes(self, *args, **kwargs):
        def deep_parse(item):
            if isinstance(item, torch.Tensor):
                return item.size()
            elif isinstance(item, (int, float, str, torch.dtype)):
                return item
            elif isinstance(item, (list, tuple)):
                return [deep_parse(sub_item) for sub_item in item]
            elif isinstance(item, dict):
                return {key: deep_parse(value) for key, value in item.items()}
            return None

        parsed_args = [deep_parse(arg) for arg in args]
        parsed_kwargs = {key: deep_parse(value) for key, value in kwargs.items()}
        return parsed_args if len(parsed_args) > 0 else parsed_kwargs

    def init_user_config(self):
        # self.device = Config.device
        self.cpu_mode = Config.cpu_mode
        self.set_dtypes(Config.user_desired_dtypes)
        self.set_metrics(Config.user_desired_metrics)
        self.set_shapes()

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

    def get_input_iter(self, dtype) -> Generator:
        # """Return the dynamic input iterator for each Operator."""
        raise NotImplementedError(
            "Each Benchmark must implement its own input iterator."
        )

    def get_inputs(self, dtype):
        if self._input_iter is None:
            self._input_iter = self.get_input_iter(dtype)
        try:
            return next(self._input_iter)
        except StopIteration:
            return None

    def run(self):
        self.init_user_config()

        def _unpack_to_args_kwargs(input_tuple: Tuple[Any, ...]):
            args = []
            kwargs = {}
            for item in input_tuple:
                if (
                    isinstance(item, torch.Tensor)
                    or isinstance(item, (int, float))
                    or item is None
                    or isinstance(item, (list, tuple))
                ):
                    args.append(item)
                elif isinstance(item, dict):
                    kwargs.update(item)
            return args, kwargs

        for dtype in self.to_bench_dtypes:
            metrics = []
            for input in self.get_input_iter(dtype):
                metric = BenchmarkMetrics()
                args, kwargs = _unpack_to_args_kwargs(input)
                shape = self.record_shapes(*args, **kwargs)
                metric.shape_detail = shape
                if "latency_base" in self.to_bench_metrics:
                    metric.latency_base = self.get_latency(
                        self.torch_op, *args, **kwargs
                    )
                if "latency" in self.to_bench_metrics:
                    if self.gems_op:
                        metric.latency = self.get_latency(self.gems_op, *args, **kwargs)
                    else:
                        with flag_gems.use_gems():
                            metric.latency = self.get_latency(
                                self.torch_op, *args, **kwargs
                            )
                if "speedup" in self.to_bench_metrics:
                    metric.speedup = metric.latency / metric.latency_base
                if "tflops" in self.to_bench_metrics:
                    metric.tflops = self.get_tflops(self.torch_op, *args, **kwargs)
                metrics.append(metric)
            result = BenchmarkResult(
                op_name=self.op_name,
                dtype=str(dtype),
                mode="cpu" if Config.cpu_mode else "cuda",
                result=metrics,
            )
            print(result)

    def legacy_run(self):
        for dtype in self.dtypes:
            metrics = []
            for size in self.shapes:
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
                mode="cpu" if self.cpu_mode else "cuda",
                result=metrics,
            )
            print(result)


class GenericBenchmark(Benchmark):

    """
    Generic benchmark for tensor operations with different types of inputs.
    """

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def set_shapes(self):
        self.shapes = self.DEFAULT_SHAPES[:]
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            # TODO: more suitable shapes
            small_shapes = [(2, 2), (1024, 1)]
            large_shapes = [
                (10240, 10240),
            ]
            self.shapes.extend(small_shapes)
            self.shapes.extend(large_shapes)

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)


def binary_input_fn(shape, cur_dtype, device):
    inp1 = torch.randn(shape, dtype=cur_dtype, device=device)
    inp2 = torch.randn(shape, dtype=cur_dtype, device=device)
    yield inp1, inp2


def unary_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    yield inp,


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
