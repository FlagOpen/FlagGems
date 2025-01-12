import gc
import importlib
import logging
import time
from typing import Any, Generator, List, Optional, Tuple

import pytest
import torch
import triton
import yaml

import flag_gems

from .attri_util import (
    BOOL_DTYPES,
    DEFAULT_METRICS,
    DEFAULT_SHAPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    BenchLevel,
    BenchmarkMetrics,
    BenchmarkResult,
    OperationAttribute,
    check_metric_dependencies,
)
from .conftest import Config

torch_backend_device = flag_gems.runtime.torch_backend_device
torch_device_fn = flag_gems.runtime.torch_device_fn
device = flag_gems.device
torch_backend_device.matmul.allow_tf32 = False


def SkipVersion(module_name, skip_pattern):
    cmp = skip_pattern[0]
    assert cmp in ("=", "<", ">"), f"Invalid comparison operator: {cmp}"
    try:
        M, N = skip_pattern[1:].split(".")
        M, N = int(M), int(N)
    except Exception:
        raise ValueError("Cannot parse version number from skip_pattern.")

    try:
        module = importlib.import_module(module_name)
        version = module.__version__
        major, minor = map(int, version.split(".")[:2])
    except Exception:
        raise ImportError(f"Cannot determine version of module: {module_name}")

    if cmp == "=":
        return major == M and minor == N
    elif cmp == "<":
        return (major, minor) < (M, N)
    else:
        return (major, minor) > (M, N)


def triton_testing_do_bench_rewritting(fn, warmup=25, rep=100, grad_to_none=None,
                                       quantiles=None, fast_flush=True, return_mode="mean",
                                       device_type="cuda", fixed_warmup_rep_runs=True):
    """
    This is a rewritten version of the original `triton.testing.do_bench` function.

    Benchmark the runtime of the provided function. By default, return the median runtime
    of :code:`fn` along with the 20-th and 80-th performance percentile.

    This function supports two modes for determining the number of warmup and repetition
    runs, by appending a parameter called `fixed_warmup_rep_runs`:
    1. Dynamic Mode (the original implementation of `triton.testing.do_bench`):
       Estimates the runtime of the kernel and dynamically adjusts the number of warmup and
       repetition runs based on the provided `warmup` and `rep` times (in milliseconds).
    2. Fixed Mode (default in this rewritten version, and consistent with torch's testing):
       Uses the provided `warmup` and `rep` values directly as the number of warmup and
       repetition runs.

    Please refer to the original implementation of `triton.testing.do_bench` function for
    more details:
    https://github.com/triton-lang/triton/blob/199fd8a239068318e94d39843c4c676f44883bd3/python/triton/testing.py#L162
    """

    assert return_mode in ["min", "max", "mean", "median"]

    di = torch._dynamo.device_interface.get_interface_for_device(device_type)

    fn()
    di.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device_type)
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device=device_type)

    if fixed_warmup_rep_runs:
        # Estimate the runtime of the function
        start_event = di.Event(enable_timing=True)
        end_event = di.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            cache.zero_()
            fn()
        end_event.record()
        di.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        # compute number of warmup and repeat
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))
    else:
        n_warmup = warmup
        n_repeat = rep

    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


class Benchmark:
    device: str = device
    DEFAULT_METRICS = DEFAULT_METRICS
    DEFAULT_DTYPES = FLOAT_DTYPES
    DEFAULT_SHAPES = DEFAULT_SHAPES
    DEFAULT_SHAPE_DESC = "M, N"
    DEFAULT_SHAPE_FILES = "core_shapes.yaml"
    """
    the base class for the operations benchmark
    """

    def __init__(
        self,
        op_name,
        torch_op,
        dtypes=None,
        is_backward=False,
        **kwargs,
    ):
        self.op_name = op_name
        if is_backward:
            self.op_name += " backward"
        self.torch_op = torch_op
        self.gems_op = None
        self.is_backward = is_backward
        self._input_iter = None

        # Theoretical supported dtypes, metrics for the operation.
        # These are set by default.
        self.dtypes = dtypes if dtypes is not None else self.DEFAULT_DTYPES
        self.metrics = self.DEFAULT_METRICS
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC
        self.shape_file = self.DEFAULT_SHAPE_FILES

        # Actual dtypes and metrics to be used in the benchmark,
        # can be influenced by user input.
        self.to_bench_dtypes = self.dtypes
        self.to_bench_metrics = self.metrics

        # additional properties
        for k in kwargs:
            if hasattr(self, k):
                setattr(self, k, kwargs[k])

    def set_metrics(self, user_desired_metrics: Optional[List[str]]):
        # Validate user-specified metrics
        if user_desired_metrics:
            invalid_metrics = [
                metric for metric in user_desired_metrics if metric not in self.metrics
            ]
            if invalid_metrics:
                raise ValueError(
                    f"Invalid metrics: {', '.join(invalid_metrics)} for operation: '{self.op_name}'"
                )
            unsatisfied_metrics = check_metric_dependencies(user_desired_metrics)
            if unsatisfied_metrics:
                raise ValueError(
                    f"Unsatisfied metric dependencies: {', '.join(unsatisfied_metrics)}"
                )

        self.to_bench_metrics = user_desired_metrics or self.metrics
        if (
            hasattr(self, "set_more_metrics")
            and callable(getattr(self, "set_more_metrics"))
            and Config.bench_level == BenchLevel.COMPREHENSIVE
            and not Config.query
        ):
            for metric in self.set_more_metrics():
                if metric not in self.to_bench_metrics:
                    self.to_bench_metrics.append(metric)

    def set_more_metrics(self):
        """Base method (optional to override in subclasses). Returns additional shapes if applicable."""
        return []

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

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        # Validate user-spicified shapes files
        import os

        if not os.path.isfile(shape_file_path):
            raise FileNotFoundError(f"Shape file '{shape_file_path}' does not exist.")
        try:
            with open(shape_file_path, "r") as file:
                yaml_config = yaml.safe_load(file)
                if self.op_name in yaml_config:
                    self.shapes = yaml_config[self.op_name].get(
                        "shapes", self.DEFAULT_SHAPES
                    )
                    self.shape_desc = yaml_config[self.op_name].get(
                        "shape_desc", self.DEFAULT_SHAPE_DESC
                    )
                else:
                    for cls in type(self).__mro__:
                        class_name = cls.__name__
                        if class_name in yaml_config:
                            self.shapes = yaml_config[class_name].get(
                                "shapes", self.DEFAULT_SHAPES
                            )
                            self.shape_desc = yaml_config[class_name].get(
                                "shape_desc", self.DEFAULT_SHAPE_DESC
                            )
                            break
                    else:
                        self.shapes = self.DEFAULT_SHAPES

            self.shapes = [tuple(shape) for shape in self.shapes]
            # merge shapes from subclass If subclass has `set_more_shapes`, call it to merge shapes
            if (
                hasattr(self, "set_more_shapes")
                and callable(getattr(self, "set_more_shapes"))
                and Config.bench_level == BenchLevel.COMPREHENSIVE
                and not Config.query
            ):
                # Merge shapes using subclass-specific logic
                additional_shapes = self.set_more_shapes()
                # self.shapes = additional_shapes
                if additional_shapes:
                    self.shapes = list(dict.fromkeys(self.shapes + additional_shapes))
        except yaml.YAMLError as e:
            raise ValueError(
                f"Shape file '{shape_file_path}' is not a valid YAML file. Error: {e}"
            )

    def set_more_shapes(self) -> Optional[List[List[int]]]:
        """Base method (optional to override in subclasses). Returns additional shapes if applicable."""
        return None

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
        if parsed_args and parsed_kwargs:
            return parsed_args, parsed_kwargs
        return parsed_args if parsed_args else parsed_kwargs

    def init_default_config(self):
        self.set_shapes(self.DEFAULT_SHAPE_FILES)

    def init_user_config(self):
        # TODO: device setting
        self.cpu_mode = Config.cpu_mode
        self.set_dtypes(Config.user_desired_dtypes)
        self.set_metrics(Config.user_desired_metrics)
        self.set_shapes(Config.shape_file)

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
            torch_device_fn.synchronize()
            start = time.time()
            for i in range(Config.repetition):
                fn()
            torch_device_fn.synchronize()
            end = time.time()
            latency = (end - start) / Config.repetition * 1000
        else:
            latency = triton_testing_do_bench_rewritting(
                fn,
                warmup=Config.warm_up,
                rep=Config.repetition,
                return_mode="median",
            )
        # average latency in ms
        return latency

    def get_gbps(self, args, latency=None):
        # """Return the dynamic input iterator for each Operator."""
        raise NotImplementedError(
            "Each Benchmark must implement its own input iterator."
        )

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

    def unpack_to_args_kwargs(self, input_tuple: Tuple[Any, ...]):
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
        if self.is_backward:
            args = [
                (
                    a.clone().requires_grad_()
                    if torch.is_tensor(a) and torch.is_floating_point(a)
                    else a
                )
                for a in args
            ]
        return args, kwargs

    def run(self):
        if Config.query:
            self.init_default_config()
            attri = OperationAttribute(
                op_name=self.op_name,
                recommended_core_shapes=self.shapes,
                shape_desc=self.shape_desc,
            )
            print(attri)
            logging.info(attri.to_dict())
            return
        self.init_user_config()
        for dtype in self.to_bench_dtypes:
            metrics = []
            for input in self.get_input_iter(dtype):
                metric = BenchmarkMetrics()
                try:
                    args, kwargs = self.unpack_to_args_kwargs(input)
                    metric.shape_detail = self.record_shapes(*args, **kwargs)
                    if "latency_base" in self.to_bench_metrics:
                        metric.latency_base = self.get_latency(
                            self.torch_op, *args, **kwargs
                        )
                    if "latency" in self.to_bench_metrics:
                        if self.gems_op:
                            metric.latency = self.get_latency(
                                self.gems_op, *args, **kwargs
                            )
                        else:
                            with flag_gems.use_gems():
                                metric.latency = self.get_latency(
                                    self.torch_op, *args, **kwargs
                                )
                    if "speedup" in self.to_bench_metrics:
                        metric.speedup = metric.latency_base / metric.latency
                    if "gbps" in self.to_bench_metrics:
                        metric.gbps_base = self.get_gbps(
                            args, latency=metric.latency_base
                        )
                        metric.gbps = self.get_gbps(args, latency=metric.latency)
                    if "tflops" in self.to_bench_metrics:
                        metric.tflops = (
                            self.get_tflops(self.torch_op, *args, **kwargs)
                            / metric.latency
                            / 1e12
                            * 1e3
                        )
                        # utilization = metric.tflops / metric.latency / 1e12 * 1e3
                except Exception as e:
                    metric.error_msg = str(e)
                    pytest.fail(str(e))  # raise exception again
                finally:
                    metrics.append(metric)
                    gc.collect()
            result = BenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode="cpu" if Config.cpu_mode else device,
                result=metrics,
            )
            print(result)
            logging.info(result.to_json())


class GenericBenchmark(Benchmark):
    """
    A generic benchmark class for most of the operations.

    This class extends the Benchmark base class. It allows users to specify custom
    input functions and shapes, making it suitable for a wide range of tensor
    operations including both unary and binary operations.

    Usage example:
        benchmark = GenericBenchmark(op_name="add", torch_op=torch.add, input_fn=binary_input_fn)
        benchmark.run()
    """

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def set_more_shapes(self):
        more_shapes_1d = [
            (2**28,),
        ]
        more_shapes_2d = [(10000, 2**i) for i in (0, 8, 16)]
        more_shapes_3d = [(100, 2**i, 100) for i in (0, 8, 16)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)


class GenericBenchmarkFilterShapes(GenericBenchmark):
    def __init__(self, exclude_dims: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exclude_dims = exclude_dims

    def set_more_shapes(self):
        shapes = super().set_more_shapes()
        if self.exclude_dims is not None:
            return [shape for shape in shapes if len(shape) != self.exclude_dims]
        return shapes


class GenericBenchmarkExcluse1D(GenericBenchmarkFilterShapes):
    """
    exclude 1d shapes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(exclude_dims=1, *args, **kwargs)


class GenericBenchmarkExcluse3D(GenericBenchmarkFilterShapes):
    """
    exclude 3d shapes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(exclude_dims=3, *args, **kwargs)


class GenericBenchmark2DOnly(GenericBenchmarkFilterShapes):
    """
    2d shapes only
    """

    def __init__(self, *args, **kwargs):
        super().__init__(exclude_dims=None, *args, **kwargs)

    def set_more_shapes(self):
        shapes = super().set_more_shapes()
        return [shape for shape in shapes if len(shape) == 2]


def generate_tensor_input(shape, dtype, device):
    if dtype in FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)
    elif dtype in INT_DTYPES:
        return torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device=device,
        )
    elif dtype in BOOL_DTYPES:
        return torch.randint(0, 2, size=shape, dtype=dtype, device=device)


def binary_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = generate_tensor_input(shape, cur_dtype, device)
    yield inp1, inp2


def unary_input_fn(shape, cur_dtype, device):
    yield generate_tensor_input(shape, cur_dtype, device),
