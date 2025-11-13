import pytest
import torch
from typing import Generator

import flag_gems
import triton

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import (
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


# 自定义的 BenchmarkResult 类 (保持不变)
class DgegluBenchmarkResult(BenchmarkResult):
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
        latency_base_str = f"{metrics.latency_base:.6f}"
        latency_str = f"{metrics.latency:.6f}"
        speedup_str = f"{metrics.speedup:.3f}"
        gbps_base_str = f"{metrics.gbps_base:.3f}"
        gbps_str = f"{metrics.gbps:.3f}"
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


@pytest.mark.skipif(not TE_AVAILABLE, reason="Transformer Engine backend is not available for reference.")
@pytest.mark.dgeglu
def test_perf_dgeglu():
    ref_op = lambda grad, inp: tex.dgeglu(grad, inp, None)
    
    # 我们假设 flag_gems.dgeglu 内部有塑形逻辑，可以处理多维张量
    # 如果它严格要求2D输入，需要在这里加一个包装器
    gems_op = flag_gems.dgeglu 
    
    def get_gbps(args: tuple, latency: float) -> float:
        grad_output, input_tensor = args
        io_amount = grad_output.numel() * grad_output.element_size() + \
                    2 * (input_tensor.numel() * input_tensor.element_size())
        return io_amount * 1e-9 / (latency * 1e-3)

    for dtype in FLOAT_DTYPES:
        results_for_dtype = []
        
        # =================================================================
        # 【核心修正 1】定义一个包含多维形状的列表
        # =================================================================
        # 形状元组的最后一个元素代表 2*N
        dgeglu_shapes = [
            (4096, 2048),              # 2D case: M=4096, N=1024
            (16, 1024, 4096),          # 3D case: B=16, S=1024, N=2048
            (8, 512, 8192),            # 3D case
            (4, 128, 8, 2048),         # 4D case
            # (2, 64, 16, 32, 1024),     # 5D case
        ]

        for shape in dgeglu_shapes:
            # =================================================================
            # 【核心修正 2】动态地、与维度无关地创建输入张量
            # =================================================================
            
            # 确保最后一个维度是偶数
            if shape[-1] % 2 != 0:
                # 在实际测试中，我们应该只提供有效的形状
                # 这里为了健壮性可以跳过或调整
                continue

            # 创建 input_tensor
            input_tensor = generate_tensor_input(shape, dtype, 'cuda')

            # 动态计算 grad_output 的形状
            grad_output_shape = list(shape)
            grad_output_shape[-1] //= 2
            grad_output = generate_tensor_input(tuple(grad_output_shape), dtype, 'cuda')
            
            inputs = (grad_output, input_tensor)

            # --- 计时和计算逻辑保持不变 ---
            ref_latency = triton.testing.do_bench(lambda: ref_op(*inputs))
            gems_latency = triton.testing.do_bench(lambda: gems_op(*inputs))
            
            speedup = ref_latency / gems_latency
            ref_gbps = get_gbps(inputs, ref_latency)
            gems_gbps = get_gbps(inputs, gems_latency)
            
            metrics = BenchmarkMetrics(
                latency_base=ref_latency,
                latency=gems_latency,
                speedup=speedup,
                gbps_base=ref_gbps,
                gbps=gems_gbps,
                shape_detail=[tuple(grad_output.shape), tuple(input_tensor.shape)],
            )
            results_for_dtype.append(metrics)

        # --- 打印逻辑保持不变 ---
        result_formatter = DgegluBenchmarkResult(
            op_name="dgeglu",
            dtype=str(dtype),
            mode="kernel",
            level=Config.bench_level.value,
            result=results_for_dtype,
        )
        print(result_formatter)