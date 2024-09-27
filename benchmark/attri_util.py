import operator
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import List, Optional, Tuple


class ReadOnly:
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value


BLAS_OPS = ReadOnly(["addmm", "mv", "addmm", "mm", "outer"])

DEFAULT_WARMUP_COUNT = 100
DEFAULT_ITER_COUNT = 1000

# BLAS situation
# BLAS shapes is defined by (B,M,N,K), it is different from the non blas Shapes
DEFAULT_BLAS_BENCH_SHAPES = [(1, 1, 1, 32), (4, 15, 160, 1024), (16, 495, 5333, 71)]
DEFAULT_BLAS_WITHOUT_BATCH_BENCH_SHAPES = [(1, 1, 32), (15, 160, 1024), (495, 5333, 71)]

# Non BLAS shapes are defined by (B, M, N) op (B, M, N)
DEFAULT_NON_BLAS_BENCH_SHAPES = [(1024, i * 64) for i in range(1, 22, 5)]


def get_recommended_shapes(
    op_name: str, op_specified_shapes: Optional[List[Tuple[int, ...]]]
):
    def _shapes_sort(shapes):
        return sorted(shapes, key=lambda x: reduce(operator.mul, x))

    if op_specified_shapes:
        return _shapes_sort(op_specified_shapes)
    shapes = DEFAULT_NON_BLAS_BENCH_SHAPES
    if op_name in ["bmm", "mv"]:
        shapes = DEFAULT_BLAS_BENCH_SHAPES
    elif op_name in ["addmm", "mm", "outer"]:
        shapes = DEFAULT_BLAS_WITHOUT_BATCH_BENCH_SHAPES
    return _shapes_sort(shapes)


class BenchLevel(Enum):
    COMPREHENSIVE = "comprehensive"
    CORE = "core"


@dataclass
class OperationAttribute:
    op_name: str
    # Recommended core benchmark shapes for given op
    recommended_core_shapes: List[Tuple[int, ...]]

    def __str__(self):
        name_str = f"Operator name                    |  {self.op_name}\n"
        if self.op_name in BLAS_OPS.value:
            shapes_str = (
                f"Recommended Core Shapes[DMNK]    |  {self.recommended_core_shapes}\n"
            )
        else:
            shapes_str = (
                f"Recommended Core Shapes[DMN]     |  {self.recommended_core_shapes}\n"
            )
        return name_str + shapes_str

    def to_dict(self):
        return self.__dict__


@dataclass
class BenckmarkMatrics:
    # the simple version shape info, this shape setted here just to with the last version.
    shape: int
    # the detailed size info
    shape_detail: Optional[Tuple[int, ...]]
    # latency_base in ms
    latency_base: Optional[float] = None
    # latency in ms
    latency: Optional[float] = None
    # speedup over baseline
    speedup: Optional[float] = None
    # Not Implemented Yet, accuracy over baseline
    accuracy: Optional[bool] = None
    # Not Implemented Yet, tflops
    tflops: Optional[float] = None
    # Not Implemented Yet, utility
    utilization: Optional[float] = None


@dataclass
class BenchmarkResult:
    "record the benchmark Result for each operator"
    # the uniq name of op
    op_name: str
    dtype: str
    mode: str
    # the Result
    result: List[BenckmarkMatrics]

    def __str__(self):
        head = (
            f"\nOperator {self.op_name}  Performance Test (dtype={self.dtype}, mode={self.mode})\n"
            f"Size      Torch Latency (ms)    Gems Latency (ms)    Gems Speedup    Size Detail\n"
            f"--------------------------------------------------------------------------------\n"
        )
        matrics = ""
        for ele in self.result:
            line = (
                f"{ele.shape: <10}"
                f"{ele.latency_base: >18.6}"
                f"{ele.latency: >21.6}"
                f"{ele.speedup: >16.3}"
                f"{' ' * 5}"
                f"{ele.shape_detail}\n"
            )
            matrics += line
        return head + matrics

    def to_dict(self):
        return self.__dict__
