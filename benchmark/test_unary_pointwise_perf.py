from typing import Generator, Tuple

import pytest
from pytest import mark
import torch
import logging
import gc
import flag_gems

from benchmark.performance_utils import (
    Benchmark,
    BenchmarkMetrics,
    BenchmarkResult,
    Config,
    generate_tensor_input,
)

try:
    from transformer_engine.pytorch import cpp_extensions as tex
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False

from benchmark.attri_util import (
    BOOL_DTYPES,
    COMPLEX_DTYPES,
    DEFAULT_METRICS,
    FLOAT_DTYPES,
    INT_DTYPES,
)
from benchmark.performance_utils import Benchmark, generate_tensor_input, vendor_name

fp64_is_supported = flag_gems.runtime.device.support_fp64


class UnaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking unary pointwise operations.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp,

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


forward_operations = [
    ("abs", torch.abs, FLOAT_DTYPES),
    ("angle", torch.angle, COMPLEX_DTYPES + [torch.float32] + INT_DTYPES + BOOL_DTYPES),
    ("erf", torch.erf, FLOAT_DTYPES),
    ("exp", torch.exp, FLOAT_DTYPES),
    ("exp2", torch.exp2, FLOAT_DTYPES),
    ("neg", torch.neg, FLOAT_DTYPES),
    ("reciprocal", torch.reciprocal, FLOAT_DTYPES),
    ("sqrt", torch.sqrt, FLOAT_DTYPES),
    ("rsqrt", torch.rsqrt, FLOAT_DTYPES),
    ("logical_not", torch.logical_not, INT_DTYPES + BOOL_DTYPES),
    ("log", torch.log, FLOAT_DTYPES),
    # ("triu", torch.triu, FLOAT_DTYPES),  # do not support 1d shapes
    # Dropout
    ("dropout", torch.nn.Dropout(p=0.5), FLOAT_DTYPES),
    # Activation operations
    ("celu", torch.nn.functional.celu, FLOAT_DTYPES),
    ("elu", torch.nn.functional.elu, FLOAT_DTYPES),
    ("gelu", torch.nn.functional.gelu, FLOAT_DTYPES),
    ("relu", torch.nn.functional.relu, FLOAT_DTYPES),
    ("softplus", torch.nn.functional.softplus, FLOAT_DTYPES),
    ("sigmoid", torch.sigmoid, FLOAT_DTYPES),
    ("log_sigmoid", torch.nn.functional.logsigmoid, FLOAT_DTYPES),
    ("silu", torch.nn.functional.silu, FLOAT_DTYPES),
    # Trigonometric operations
    ("cos", torch.cos, FLOAT_DTYPES),
    ("sin", torch.sin, FLOAT_DTYPES),
    ("tanh", torch.tanh, FLOAT_DTYPES),
    ("atan", torch.atan, FLOAT_DTYPES),
    # Bitwise operations
    ("bitwise_not", torch.bitwise_not, INT_DTYPES),
    # Numerical Checks
    ("isinf", torch.isinf, FLOAT_DTYPES),
    ("isnan", torch.isnan, FLOAT_DTYPES),
    ("isfinite", torch.isfinite, FLOAT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name, None),
        )
        for name, op, dtype in forward_operations
    ],
)
def test_general_unary_pointwise_perf(op_name, torch_op, dtypes):
    if vendor_name == "mthreads" and op_name == "angle":
        pytest.skip(" Unsupport complex dtype")
    bench = UnaryPointwiseBenchmark(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
    bench.run()


forward_inplace_operations = [
    ("abs_", torch.abs_, FLOAT_DTYPES),
    # ("angle", torch.angle, COMPLEX_DTYPES + [torch.float32] + INT_DTYPES + BOOL_DTYPES),
    ("erf_", torch.erf_, FLOAT_DTYPES),
    ("exp_", torch.exp_, FLOAT_DTYPES),
    ("exp2_", torch.exp2_, FLOAT_DTYPES),
    ("neg_", torch.neg_, FLOAT_DTYPES),
    ("reciprocal_", torch.reciprocal_, FLOAT_DTYPES),
    ("sqrt_", torch.sqrt_, FLOAT_DTYPES),
    ("rsqrt_", torch.rsqrt_, FLOAT_DTYPES),
    # Activation operations
    ("celu_", torch.nn.functional.celu_, FLOAT_DTYPES),
    ("elu_", torch.nn.functional.elu_, FLOAT_DTYPES),
    ("gelu_", torch.ops.aten.gelu_.default, FLOAT_DTYPES),
    ("relu_", torch.relu_, FLOAT_DTYPES),
    ("sigmoid_", torch.sigmoid_, FLOAT_DTYPES),
    ("silu_", lambda a: torch.nn.functional.silu(a, inplace=True), FLOAT_DTYPES),
    # Trigonometric operations
    ("cos_", torch.cos_, FLOAT_DTYPES),
    ("sin_", torch.sin_, FLOAT_DTYPES),
    ("tanh_", torch.tanh_, FLOAT_DTYPES),
    # Bitwise operations
    ("bitwise_not_", lambda a: a.bitwise_not_(), INT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name, None),
        )
        for name, op, dtype in forward_inplace_operations
    ],
)
def test_general_inplace_unary_pointwise_perf(op_name, torch_op, dtypes):
    bench = UnaryPointwiseBenchmark(
        op_name=op_name, torch_op=torch_op, dtypes=dtypes, is_inplace=True
    )
    bench.run()


backward_operations = [
    ("gelu", torch.nn.functional.gelu, FLOAT_DTYPES),
]


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            name,
            op,
            dtype,
            marks=getattr(pytest.mark, name + "_backward", None),
        )
        for name, op, dtype in backward_operations
    ],
)
def test_general_unary_pointwise_backward_perf(op_name, torch_op, dtypes):
    bench = UnaryPointwiseBenchmark(
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
        is_backward=True,
    )
    bench.run()


class ToDtypeBenchmark(UnaryPointwiseBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=torch.float32, device=self.device)
            yield inp, cur_dtype


@pytest.mark.to
def test_to_dtype_perf():
    bench = ToDtypeBenchmark(
        op_name="to",
        torch_op=torch.Tensor.to,
        dtypes=[torch.float16, torch.bfloat16]
        + ([torch.float64] if fp64_is_supported else []),
    )
    bench.run()


class EluBackwardBenchmark(UnaryPointwiseBenchmark):
    def get_input_iter(self, cur_dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            grad_out = torch.randn_like(inp)
            alpha = 1.0
            scale = 1.0
            input_scale = 1.0
            is_result = False

            yield grad_out, alpha, scale, input_scale, is_result, inp


@pytest.mark.elu_backward
def test_elu_backward_perf():
    bench = EluBackwardBenchmark(
        op_name="elu_backward",
        torch_op=torch.ops.aten.elu_backward,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class GluBenchmark(UnaryPointwiseBenchmark):
    # Glu test requires even numbers
    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(1, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(1, 15, 4)]
        return special_shapes_2d + sp_shapes_3d


@pytest.mark.glu
def test_glu_perf():
    bench = GluBenchmark(
        op_name="glu",
        torch_op=torch.nn.functional.glu,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.glu_backward
def test_glu_backward_perf():
    bench = GluBenchmark(
        op_name="glu",
        torch_op=torch.nn.functional.glu,
        dtypes=FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()


class BinaryPointwiseBenchmark(Benchmark):
    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp1 = generate_tensor_input(shape, cur_dtype, self.device)
            shift_amount = torch.randint(
                0, 8, shape, dtype=cur_dtype, device=self.device
            )
            yield inp1, shift_amount


@pytest.mark.bitwise_left_shift
def test_bitwise_left_shift_perf():
    bench = BinaryPointwiseBenchmark(
        op_name="bitwise_left_shift",
        torch_op=torch.bitwise_left_shift,
        dtypes=INT_DTYPES,
    )
    bench.run()


@pytest.mark.bitwise_right_shift
def test_bitwise_right_shift_perf():
    bench = BinaryPointwiseBenchmark(
        op_name="bitwise_right_shift",
        torch_op=torch.bitwise_right_shift,
        dtypes=INT_DTYPES,
    )
    bench.run()

class SwigluBenchmarkResult(BenchmarkResult):
    """自定义的结果格式化类，用于 swiglu 测试的特定输出样式。"""
    def __str__(self) -> str:
        header_title = (
            f"\nOperator: {self.op_name}  Performance Test (dtype={self.dtype}, mode={self.mode},"
            f"level={self.level})\n"
        )
        col_names = [
            f"{'Status':<12}",
            f"{'TE Latency (ms)':>20}",
            f"{'Gems Latency (ms)':>20}",
            f"{'Gems Speedup':>20}",
            f"{'TE GBPS':>20}",
            f"{'Gems GBPS':>20}",
            "          Size Detail",
        ]
        
        header_col_names = "".join(col_names)
        header_break = "\n" + "-" * (len(header_col_names) + 10)
        header = header_title + header_col_names + header_break

        metrics_lines = "".join(self._format_metrics(ele) for ele in self.result)
        return header + metrics_lines

    def _format_metrics(self, metrics: BenchmarkMetrics) -> str:
        status = "SUCCESS" if metrics.error_msg is None else "FAILED"
        latency_base_str = f"{metrics.latency_base:.6f}" if metrics.latency_base is not None else "N/A"
        latency_str = f"{metrics.latency:.6f}" if metrics.latency is not None else "N/A"
        speedup_str = f"{metrics.speedup:.3f}" if metrics.speedup is not None else "N/A"
        gbps_base_str = f"{metrics.gbps_base:.3f}" if metrics.gbps_base is not None else "N/A"
        gbps_str = f"{metrics.gbps:.3f}" if metrics.gbps is not None else "N/A"
        shape_detail_str = f"{metrics.shape_detail}"

        data_line = (
            f"\n{status:<12}"
            f"{latency_base_str:>20}"
            f"{latency_str:>20}"
            f"{speedup_str:>20}"
            f"{gbps_base_str:>20}"
            f"{gbps_str:>20}"
            f"          {shape_detail_str}"
        )
        return data_line


class SwigluForwardBenchmark(Benchmark):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = DEFAULT_METRICS + ["gbps"]
        self.to_bench_metrics = self.metrics

    def set_shapes(self, shape_file_path: str = None):
        """定义 Swiglu 前向测试的形状列表（最后一维必须为偶数）"""
        core_shapes = [
            (1024, 1024),           # 最后一维1024（偶数）
            (4096, 2048),           # 最后一维2048（偶数）
            (16, 1024, 4096),       # 最后一维4096（偶数）
            (8, 512, 8192),         # 最后一维8192（偶数）
            (4, 128, 8, 2048),      # 最后一维2048（偶数）
        ]
        self.shapes = core_shapes
        
        if Config.bench_level.value == 'comprehensive':
            additional_shapes = self.set_more_shapes()
            if additional_shapes:
                self.shapes = list(dict.fromkeys(self.shapes + additional_shapes))

    def set_more_shapes(self):
        """为全面测试模式定义额外形状（确保最后一维为偶数）"""
        special_shapes_2d = [(4096, 2**i) for i in range(8, 14, 2)]  # 2^8=256, 2^10=1024, etc.（均为偶数）
        sp_shapes_3d = [(16, 1024, 2**i) for i in range(10, 15, 2)]  # 最后一维为偶数
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype: torch.dtype) -> Generator:
        """生成 Swiglu 前向计算的输入张量（确保最后一维为偶数）"""
        for input_shape in self.shapes:
            # 校验最后一维是否为偶数（增强健壮性）
            if input_shape[-1] % 2 != 0:
                raise ValueError(f"Swiglu forward input shape {input_shape} has odd last dimension")
            input_tensor = generate_tensor_input(input_shape, cur_dtype, self.device)
            yield (input_tensor,)

    def get_gbps(self, args: tuple, latency: float) -> float:
        if not latency or latency == 0:
            return 0.0
        input_tensor, = args
        element_size = input_tensor.element_size()
        # 前向计算：读取input_tensor（2*hidden_dim），输出hidden_dim，总数据量为 input + output
        total_bytes = (input_tensor.numel() + (input_tensor.numel() // 2)) * element_size
        return total_bytes / (latency * 1e6)  # 转换为 GB/s

    def run(self):
        if Config.query:
            super().run()
            return
            
        self.init_user_config()
        if 'gbps' not in self.to_bench_metrics and any(m in self.to_bench_metrics for m in ['latency', 'latency_base']):
            self.to_bench_metrics.append('gbps')

        for dtype in self.to_bench_dtypes:
            metrics_list = []
            for input_data in self.get_input_iter(dtype):
                metric = BenchmarkMetrics()
                try:
                    args, kwargs = self.unpack_to_args_kwargs(input_data)
                    metric.shape_detail = self.record_shapes(*args, **kwargs)
                    
                    if "latency_base" in self.to_bench_metrics:
                        metric.latency_base = self.get_latency(self.torch_op, *args, **kwargs)
                    
                    if "latency" in self.to_bench_metrics:
                        if not self.gems_op:
                            raise ValueError("GEMS operator not set. Use bench.set_gems().")
                        metric.latency = self.get_latency(self.gems_op, *args, **kwargs)
                    
                    if "speedup" in self.to_bench_metrics and metric.latency is not None and metric.latency > 0:
                        metric.speedup = metric.latency_base / metric.latency
                    
                    if "gbps" in self.to_bench_metrics:
                        metric.gbps_base = self.get_gbps(args, latency=metric.latency_base)
                        metric.gbps = self.get_gbps(args, latency=metric.latency)
                        
                except Exception as e:
                    metric.error_msg = str(e)
                    print(f"\nBenchmark failed for shape {metric.shape_detail}: {e}")
                finally:
                    metrics_list.append(metric)
                    gc.collect()

            if not metrics_list:
                continue
                
            result_formatter = SwigluBenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=Config.mode.value,
                result=metrics_list,
            )
            print(result_formatter)
            logging.info(result_formatter.to_json())


class SwigluBackwardBenchmark(Benchmark):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = DEFAULT_METRICS + ["gbps"]
        self.to_bench_metrics = self.metrics

    def set_shapes(self, shape_file_path: str = None):

        core_shapes = [
            (1024, 1024),          
            (4096, 2048),           
            (16, 1024, 4096),      
            (8, 512, 8192),         
            (4, 128, 8, 2048),     
        ]
        self.shapes = core_shapes
        
        if Config.bench_level.value == 'comprehensive':
            additional_shapes = self.set_more_shapes()
            if additional_shapes:
                self.shapes = list(dict.fromkeys(self.shapes + additional_shapes))

    def set_more_shapes(self):
        special_shapes_2d = [(4096, 2**i) for i in range(8, 14, 2)]  
        sp_shapes_3d = [(16, 1024, 2**i) for i in range(10, 15, 2)]  
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype: torch.dtype) -> Generator:

        for input_shape in self.shapes:
     
            if input_shape[-1] % 2 != 0:
                raise ValueError(f"Swiglu backward input shape {input_shape} has odd last dimension")
            
            input_tensor = generate_tensor_input(input_shape, cur_dtype, self.device)
            
            grad_output_shape = list(input_shape)
            grad_output_shape[-1] = input_shape[-1] // 2  
            grad_output = generate_tensor_input(tuple(grad_output_shape), cur_dtype, self.device)
            
            yield (grad_output, input_tensor)

    def get_gbps(self, args: tuple, latency: float) -> float:
        if not latency or latency == 0:
            return 0.0
        grad_output, input_tensor = args
        element_size = grad_output.element_size()

        total_bytes = (grad_output.numel() + input_tensor.numel() + input_tensor.numel()) * element_size
        return total_bytes / (latency * 1e6)  

    def run(self):
        if Config.query:
            super().run()
            return
            
        self.init_user_config()
        if 'gbps' not in self.to_bench_metrics and any(m in self.to_bench_metrics for m in ['latency', 'latency_base']):
            self.to_bench_metrics.append('gbps')

        for dtype in self.to_bench_dtypes:
            metrics_list = []
            for input_data in self.get_input_iter(dtype):
                metric = BenchmarkMetrics()
                try:
                    args, kwargs = self.unpack_to_args_kwargs(input_data)
                    metric.shape_detail = self.record_shapes(*args, **kwargs)
                    
                    if "latency_base" in self.to_bench_metrics:
                        metric.latency_base = self.get_latency(self.torch_op, *args, **kwargs)
                    
                    if "latency" in self.to_bench_metrics:
                        if not self.gems_op:
                            raise ValueError("GEMS operator not set. Use bench.set_gems().")
                        metric.latency = self.get_latency(self.gems_op, *args, **kwargs)
                    
                    if "speedup" in self.to_bench_metrics and metric.latency is not None and metric.latency > 0:
                        metric.speedup = metric.latency_base / metric.latency
                    
                    if "gbps" in self.to_bench_metrics:
                        metric.gbps_base = self.get_gbps(args, latency=metric.latency_base)
                        metric.gbps = self.get_gbps(args, latency=metric.latency)
                        
                except Exception as e:
                    metric.error_msg = str(e)
                    print(f"\nBenchmark failed for shape {metric.shape_detail}: {e}")
                finally:
                    metrics_list.append(metric)
                    gc.collect()

            if not metrics_list:
                continue
                
            result_formatter = SwigluBenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=Config.mode.value,
                result=metrics_list,
            )
            print(result_formatter)
            logging.info(result_formatter.to_json())


@mark.skipif(not TE_AVAILABLE, reason="Transformer Engine backend is not available for reference.")
@mark.swiglu
def test_swiglu_forward_perf():

    bench = SwigluForwardBenchmark(
        op_name="swiglu_forward",
        torch_op=lambda x: tex.swiglu(x, None),  
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.swiglu)  
    bench.run()


@mark.skipif(not TE_AVAILABLE, reason="Transformer Engine backend is not available for reference.")
@mark.swiglu
def test_swiglu_backward_perf():

    bench = SwigluBackwardBenchmark(
        op_name="swiglu_backward",

        torch_op=lambda grad_output, input_tensor: tex.dswiglu(grad_output, input_tensor, None),
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.dswiglu)  
    bench.run()