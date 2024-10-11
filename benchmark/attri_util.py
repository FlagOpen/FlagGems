import operator
from dataclasses import dataclass, fields
from enum import Enum
from functools import reduce
from typing import List, Optional, Tuple

import torch


class ReadOnly:
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value


BLAS_OPS = ReadOnly(["addmm", "bmm", "mm", "mv", "outer", "matmul", "linear"])

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32]
BOOL_DTYPES = [
    torch.bool,
]

DEFAULT_WARMUP_COUNT = 1000
DEFAULT_ITER_COUNT = 100

# Default shapes settings
# LEGACY_SHAPES are maintained for legacy benchmark SIZE settings and may be removed in the future.
# Do not reference this elsewhere.
LEGACY_SHAPES = [i * 64 for i in range(1, 22, 5)]
# Non-BLAS shapes are currently defined as (M, N) for backward compatibility
# but will change to (B, M, N) in the future.
DEFAULT_NON_BLAS_BENCH_SHAPES = [(1024, shape) for shape in LEGACY_SHAPES]
# BLAS shapes are defined as (B, M, N, K) or (M, N, K), differing from non-BLAS shapes.
DEFAULT_BMNK_BLAS = [(16, shape, shape, shape) for shape in LEGACY_SHAPES]
DEFAULT_MNK_BLAS = [(shape, shape, shape) for shape in LEGACY_SHAPES]
# GROUP_NORM shapes are defined as (N, C, H, W, num_groups)
DEFAULT_GROUPNORM_SHAPES = [
    (16, 16, 8, 8, 16),
    (16, 16, 8, 48, 16),
    (16, 16, 8, 88, 16),
    (16, 16, 8, 128, 16),
    (16, 16, 8, 168, 16),
]

DEFAULT_BATCH = 1


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
    # Accuracy over baseline (not implemented yet)
    accuracy: Optional[float] = None
    # TFLOPS (not implemented yet)
    tflops: Optional[float] = None
    # Utilization (not implemented yet)
    utilization: Optional[float] = None


ALL_AVAILABLE_METRICS = set(map(lambda x: x.name, fields(BenchmarkMetrics))) - {
    "legacy_shape",
    "shape_detail",
}

DEFAULT_METRICS = [
    metric
    for metric in ["latency_base", "latency", "speedup"]
    if metric in ALL_AVAILABLE_METRICS
]


def get_recommended_shapes(
    op_name: str, op_specified_shapes: Optional[List[Tuple[int, ...]]]
):
    def _shapes_sort(shapes):
        return sorted(shapes, key=lambda x: reduce(operator.mul, x))

    if op_specified_shapes:
        # TODO: handle situation that list as the basic element in shape.
        return _shapes_sort(op_specified_shapes)

    shapes = DEFAULT_NON_BLAS_BENCH_SHAPES
    if op_name in ["bmm", "mv"]:
        shapes = DEFAULT_BMNK_BLAS
    elif op_name in ["addmm", "mm", "outer"]:
        shapes = DEFAULT_MNK_BLAS
    return _shapes_sort(shapes)


class BenchLevel(Enum):
    COMPREHENSIVE = "comprehensive"
    CORE = "core"


@dataclass
class OperationAttribute:
    op_name: str
    # Recommended core benchmark shapes for the given operation
    recommended_core_shapes: List[Tuple[int, ...]]

    def __str__(self) -> str:
        shapes_type = "(B),M,N,K" if self.op_name in BLAS_OPS.value else "(B),M,N"
        return (
            f"{'Operator name':<40} |  {self.op_name}\n"
            f"{'Recommended Core Shapes[' + shapes_type + ']':<40} |  {self.recommended_core_shapes}\n"
        )

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class BenchmarkResult:
    """Record the benchmark result for each operator."""

    # Unique name of the operator
    op_name: str
    dtype: str
    mode: str
    # Benchmark results
    result: List[BenchmarkMetrics]

    def __str__(self) -> str:
        header = (
            f"\nOperator: {self.op_name}  Performance Test (dtype={self.dtype}, mode={self.mode})\n"
            f"{'Size':<10} {'Torch Latency (ms)':>20} {'Gems Latency (ms)':>20} {'Gems Speedup':>20}"
            f"{'Size Detail':>20}\n"
            f"{'-' * 90}\n"
        )
        metrics_lines = "".join(self._format_metrics(ele) for ele in self.result)
        return header + metrics_lines

    def _format_metrics(self, metrics: BenchmarkMetrics) -> str:
        self.gen_legacy_shape(metrics)
        legacy_shape_str = (
            metrics.legacy_shape if metrics.legacy_shape is not None else "N/A"
        )
        latency_base_str = (
            f"{metrics.latency_base:.6f}" if metrics.latency_base is not None else "N/A"
        )
        latency_str = f"{metrics.latency:.6f}" if metrics.latency is not None else "N/A"
        speedup_str = f"{metrics.speedup:.3f}" if metrics.speedup is not None else "N/A"
        shape_detail_str = (
            metrics.shape_detail if metrics.shape_detail is not None else "N/A"
        )
        return (
            f"{legacy_shape_str:<10}"
            f"{latency_base_str:>20}"
            f"{latency_str:>20}"
            f"{speedup_str:>20}"
            f"{' ' * 10}"
            f"{shape_detail_str}\n"
        )

    def gen_legacy_shape(self, metrics: BenchmarkMetrics) -> Optional[int]:
        legacy_shapes_4d = [(16, shape, shape, shape) for shape in LEGACY_SHAPES] + [
            (1, shape, shape, shape) for shape in LEGACY_SHAPES
        ]
        legacy_shapes_2d = [(1024, shape) for shape in LEGACY_SHAPES]

        if self.op_name in BLAS_OPS.value and metrics.shape_detail in legacy_shapes_4d:
            metrics.legacy_shape = metrics.shape_detail[-1]
        elif metrics.shape_detail in legacy_shapes_2d:
            metrics.legacy_shape = metrics.shape_detail[-1]
        else:
            metrics.legacy_shape = None

    def to_dict(self) -> dict:
        return self.__dict__
