import time
import os
import torch
import triton

import flag_gems
import torch_xla.core.xla_model as xm

from .conftest import CPU_MODE
from dataclasses import asdict, dataclass, fields
from typing import List, Optional, Tuple
from .conftest import Config
import logging

WARMUP = 100
REPETITION = 1000

if os.getenv("TRITON_TX8BE_E2E_BACKEND"):
    device = "cpu"
    os.environ["TRITON_TX8BE_E2E_LOG"] = "0"
else:
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"

def custom_json_encoder(obj):
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

DEFAULT_METRICS = [
    metric
    for metric in ["latency_base", "latency", "speedup"]
]

@dataclass
class BenchmarkMetrics:
    # Legacy shape information for backward compatibility
    # This field corresponds to the 'size' field in the previous version's benchmark.
    legacy_shape: Optional[int] = None
    # Detailed size info
    shape_detail: Optional[Tuple[int, ...]] = None
    # Latency base in ms
    latency_base: Optional[float] = None
    # Latency in ms
    latency: Optional[float] = None
    # Speedup over baseline
    speedup: Optional[float] = None
    # TFLOPS (not implemented yet)
    tflops: Optional[float] = None

@dataclass
class BenchmarkResult:
    """Record the benchmark result for each operator."""

    # Unique name of the operator
    op_name: str
    dtype: str
    mode: str
    level: str
    # Benchmark results
    result: List[BenchmarkMetrics]

    def __str__(self) -> str:
        header_title = (
            f"\nOperator: {self.op_name}  Performance Test (dtype={self.dtype}, mode={self.mode},"
            f"level={self.level})\n"
        )
        col_names = [
            f"{'Status':<10}",
            f"{'Torch Latency (ms)':>20}",
            f"{'Gems Latency (ms)':>20}",
            f"{'Gems Speedup':>20}",
        ]
        col_names.append(f"{'Size Detail':>20}\n")
        header_col_names = " ".join(col_names)
        header_break = "-" * len(header_col_names) + "\n"
        header = header_title + header_col_names + header_break

        metrics_lines = "".join(self._format_metrics(ele) for ele in self.result)
        return header + metrics_lines

    def _format_metrics(self, metrics: BenchmarkMetrics) -> str:
        # self.gen_legacy_shape(metrics)
        # legacy_shape_str = (
        #     metrics.legacy_shape if metrics.legacy_shape is not None else "N/A"
        # )
        latency_base_str = (
            f"{metrics.latency_base:.6f}" if metrics.latency_base is not None else "N/A"
        )
        latency_str = f"{metrics.latency:.6f}" if metrics.latency is not None else "N/A"
        speedup_str = f"{metrics.speedup:.3f}" if metrics.speedup is not None else "N/A"


        shape_detail_str = (
            metrics.shape_detail if metrics.shape_detail is not None else "N/A"
        )
        status = "SUCCESS"
        data_line = (
            f"{status:<10}"
            f"{latency_base_str:>20}"
            f"{latency_str:>20}"
            f"{speedup_str:>20}"
        )

        data_line += " " * 10
        data_line += f"{shape_detail_str}\n"
        return data_line

    def gen_legacy_shape(self, metrics: BenchmarkMetrics) -> Optional[int]:
        first_shape = (
            metrics.shape_detail[0] if isinstance(metrics.shape_detail, list) else None
        )
        to_record_shape = (
            tuple(first_shape) if isinstance(first_shape, torch.Size) else None
        )

        if to_record_shape in LEGACY_NON_BLAS_SHAPES:
            metrics.legacy_shape = to_record_shape[-1]
        elif (
            isinstance(to_record_shape, tuple)
            and len(to_record_shape) == 2
            and to_record_shape[0] == 1024
        ):
            metrics.legacy_shape = to_record_shape[-1]
        else:
            metrics.legacy_shape = None

    def to_json(self) -> str:
        import json

        # Convert to dict and handle tuple serialization for shape_detail
        result_dict = asdict(self)
        return json.dumps(result_dict, default=custom_json_encoder)

    def to_dict(self) -> dict:
        return self.__dict__




class Benchmark:
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
        need_dim=False,
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
        self.need_dim=need_dim

    def set_gems(self, gems_op):
        self.gems_op = gems_op
        
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
    
    def profile(self, op, *args, **kwargs):
        mode = 1
        if args[0].device.type == "cpu":
            mode = 0
        fn = lambda: op(*args, **kwargs)
        if self.is_backward:
            out = fn()
            dout = torch.randn_like(out)
            fn = lambda: out.backward(dout, retain_graph=True)
            if mode != 0:
                xm.mark_step()
        if CPU_MODE:
            for i in range(WARMUP):
                out = fn()
                if mode != 0:
                    xm.mark_step()
            # torch.cuda.synchronize()
            start = time.time()
            for i in range(REPETITION):
                out = fn()
                if mode != 0:
                    xm.mark_step()
            # torch.cuda.synchronize()
            end = time.time()
            latency = (end - start) / REPETITION * 1000
        else:
            if mode == 0:
                latency = triton.testing.do_bench(
                    fn,
                    warmup=WARMUP,
                    rep=REPETITION,
                    return_mode="median",
                )
            else:
                latency = triton.testing.do_bench(
                    fn,
                    warmup=WARMUP,
                    rep=REPETITION,
                    return_mode="median",
                    device_type="xla"
                )
        # average latency in ms
        return latency

    def run(self):
        for dtype in self.dtypes:
            print(f"Operator {self.op_name} Performance Test ({dtype})")
            print("Size        Torch Latency (ms)   Gems Latency (ms)")
            print("--------------------------------------------------")
            metrics = []
            for size in self.sizes:
                args = ()
                args_xla = ()
                metric = BenchmarkMetrics()
                if self.arg_func is not None:
                    args = self.arg_func(dtype, self.batch, size)
                if self.is_backward:
                    args = tuple(
                        a.clone().requires_grad_()
                        if torch.is_tensor(a) and torch.is_floating_point(a)
                        else a
                        for a in args
                    )
                if self.need_dim:
                    args = args + (1,)

                kwargs = {}
                if self.kwargs_func is not None:
                    kwargs = self.kwargs_func(dtype, self.batch, size)

                args_xla = tuple(xm.send_cpu_data_to_device(arg, xm.xla_device()) for arg in args)
                torch_perf = self.profile(self.torch_op, *args_xla, **kwargs)
                
                if self.gems_op:
                    gems_perf = self.profile(self.gems_op, *args, **kwargs)
                else:
                    with flag_gems.use_gems():
                        gems_perf = self.profile(self.torch_op, *args, **kwargs)
                        
                metric.latency = gems_perf        
                if self.get_tflops is not None:
                    metric.tflops = (
                        self.get_tflops(self.torch_op, *args, **kwargs)
                        / metric.latency
                        / 1e12
                        * 1e3
                    )   
                                         
                metric.shape_detail = self.record_shapes(*args, **kwargs)
                metric.latency_base = torch_perf
                metric.latency = gems_perf
                metric.speedup = metric.latency_base / metric.latency
                metrics.append(metric)
                print(f"{size: <10}{torch_perf: >20.6}{gems_perf: >20.6}")
                
            result = BenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode="Tx8",
                result=metrics,
            )
            print(result)
            logging.info(result.to_json())
            
            
            
FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32]


DEFAULT_BATCH = 1
POINTWISE_BATCH = 1024
REDUCTION_BATCH = 1024
BLAS_BATCH = 16
SIZES = [i * 1024 for i in range(1, 81, 5)]
LEGACY_NON_BLAS_SHAPES = [(1024, shape) for shape in SIZES]

def unary_arg(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device=device)
    return (inp,)


def unary_int_arg(dtype, batch, size):
    inp = torch.randint(
        low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device=device
    )
    return (inp,)


def binary_args(dtype, batch, size):
    inp1 = torch.randn([batch, size], dtype=dtype, device=device)
    inp2 = torch.randn([batch, size], dtype=dtype, device=device)
    return inp1, inp2


def binary_int_args(dtype, batch, size):
    inp1 = torch.randint(
        low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device=device
    )
    inp2 = torch.randint(
        low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device=device
    )
    return inp1, inp2


def ternary_args(dtype, batch, size):
    inp1 = torch.randn([batch, size], dtype=dtype, device=device)
    inp2 = torch.randn([batch, size], dtype=dtype, device=device)
    inp3 = torch.randn([batch, size], dtype=dtype, device=device)
    return inp1, inp2, inp3
