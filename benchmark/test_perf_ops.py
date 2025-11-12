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

# 尝试导入 Transformer Engine 的 GEGLU 后端
try:
    from transformer_engine.pytorch import cpp_extensions as tex
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


class GegluBenchmarkResult(BenchmarkResult):
    def __str__(self) -> str:
        header_title = (
            f"\nOperator: {self.op_name}  Performance Test (dtype={self.dtype}, mode={self.mode},"
            f" level={self.level})\n"
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
@pytest.mark.geglu
def test_perf_geglu():
    """
    性能测试：对比 flag_gems.geglu 与 TE geglu。
    支持任意维度输入张量。
    """
    ref_op = lambda x: tex.geglu(x, None)
    gems_op = flag_gems.geglu

    def get_gbps(input_tensor, latency: float) -> float:
        io_amount = input_tensor.numel() * input_tensor.element_size() + \
                    input_tensor.numel() * input_tensor.element_size()  # 读取两次（x_a 和 x_b）
        return io_amount * 1e-9 / (latency * 1e-3)

    for dtype in FLOAT_DTYPES:
        results_for_dtype = []

        geglu_shapes = [
            (4096, 2048),             # 2D case: N=1024
            (16, 1024, 4096),         # 3D case
            (8, 512, 8192),           # 3D case
            (4, 128, 8, 2048),        # 4D case
        ]

        for shape in geglu_shapes:
            if shape[-1] % 2 != 0:
                continue  # 最后一维必须偶数

            input_tensor = generate_tensor_input(shape, dtype, 'cuda')
            inputs = (input_tensor, )

            # --- 计时 ---
            ref_latency = triton.testing.do_bench(lambda: ref_op(*inputs))
            gems_latency = triton.testing.do_bench(lambda: gems_op(*inputs))

            speedup = ref_latency / gems_latency
            ref_gbps = get_gbps(input_tensor, ref_latency)
            gems_gbps = get_gbps(input_tensor, gems_latency)

            metrics = BenchmarkMetrics(
                latency_base=ref_latency,
                latency=gems_latency,
                speedup=speedup,
                gbps_base=ref_gbps,
                gbps=gems_gbps,
                shape_detail=[tuple(input_tensor.shape)],
            )
            results_for_dtype.append(metrics)

        result_formatter = GegluBenchmarkResult(
            op_name="geglu",
            dtype=str(dtype),
            mode="kernel",
            level=Config.bench_level.value,
            result=results_for_dtype,
        )
        print(result_formatter)