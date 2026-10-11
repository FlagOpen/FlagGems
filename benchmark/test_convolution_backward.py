import csv
import json
import os
import tempfile
from functools import partial
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts
from .conftest import Config


@pytest.fixture(autouse=True)
def _strict_convolution_precision():
    if flag_gems.vendor_name == "ascend":
        import torch_npu

        # Match the IEEE FP32 Gems path instead of timing native HF32 math.
        with torch_npu.npu.aclnn.flags(allow_hf32=False):
            yield
    else:
        yield


def _ascend_kernel_latency(fn, metadata):
    import torch_npu

    # Older vendor do_bench_npu averages individual rows and removes every
    # ReduceSum kernel. Convolution backward is a multi-kernel operation whose
    # bias gradient itself uses ReduceSum, so profile complete invocations.
    warmup, repetitions = 5, 30
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()
    profile_root = os.getenv("FLAGGEMS_CONV_PROFILE_DIR")
    with tempfile.TemporaryDirectory(dir=profile_root) as directory:
        config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=False,
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(directory),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=config,
        ):
            for _ in range(repetitions):
                fn()
                torch.npu.synchronize()
        files = list(Path(directory).rglob("kernel_details.csv"))
        if len(files) != 1:
            raise RuntimeError("Expected exactly one NPU kernel profile")
        with files[0].open() as source:
            rows = list(csv.DictReader(source))
        if not rows:
            raise RuntimeError("NPU profiler returned no kernel timings")
        samples = [float(row["Duration(us)"]) for row in rows]
        if profile_root:
            with (Path(profile_root) / "samples.jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "case": metadata,
                            "repetitions": repetitions,
                            "kernels": [row["Name"] for row in rows],
                            "duration_us": samples,
                        }
                    )
                    + "\n"
                )
        return sum(samples) / repetitions / 1000


class ConvolutionBackwardBenchmark(base.GenericBenchmark):
    # batch, input channels, output channels, spatial extent, groups
    DEFAULT_SHAPES = [
        (4, 16, 32, (128,), 1),
        (4, 16, 32, (32, 32), 1),
        (2, 8, 16, (12, 12, 12), 1),
        (4, 16, 32, (32, 32), 2),
        (4, 16, 32, (32, 32), 16),
        (2, 3, 5, (17, 19), 1),
    ]

    def __init__(self, case_shape, **kwargs):
        self.case_shape = case_shape
        super().__init__(**kwargs)

    def get_latency(self, op, *args, **kwargs):
        if flag_gems.vendor_name == "ascend" and Config.mode == consts.BenchMode.KERNEL:
            fn, _ = self._benchmark_callable(op, *args, **kwargs)
            return _ascend_kernel_latency(
                fn,
                {
                    "op": str(op),
                    "shape": self.case_shape,
                    "dtype": str(args[0].dtype),
                    "transposed": args[7],
                    "mask": args[-1],
                },
            )
        return super().get_latency(op, *args, **kwargs)

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        yield from self.input_fn(self.case_shape, dtype, self.device)


def _inputs(shape, dtype, device, *, transposed, mask, out):
    batch, ci, co, spatial, groups = shape
    dim = len(spatial)
    k = (3,) * dim
    x = torch.randn((batch, ci, *spatial), device=device, dtype=dtype)
    ws = (ci, co // groups, *k) if transposed else (co, ci // groups, *k)
    w = torch.randn(ws, device=device, dtype=dtype)
    go = torch.randn((batch, co, *spatial), device=device, dtype=dtype)
    args = (
        go,
        x,
        w,
        [co],
        [1] * dim,
        [1] * dim,
        [1] * dim,
        transposed,
        [0] * dim,
        groups,
        mask,
    )
    if out:
        yield (
            *args,
            {
                "out0": torch.empty_like(x),
                "out1": torch.empty_like(w),
                "out2": torch.empty((co,), device=device, dtype=dtype),
            },
        )
    else:
        yield args


DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
MASKS = [
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, True),
]


def _cases(out):
    for shape_index, shape in enumerate(ConvolutionBackwardBenchmark.DEFAULT_SHAPES):
        for transposed in (False, True):
            for dtype in DTYPES:
                for mask, name in zip(MASKS, ("dx", "dw", "db", "all")):
                    marks = [
                        pytest.mark.skipif(
                            dtype == torch.float64
                            and not flag_gems.runtime.device.support_fp64,
                            reason="The target backend does not support the required FP64 arithmetic",
                        ),
                        pytest.mark.skipif(
                            out and not all(mask),
                            reason="ATen generated out cannot copy an undefined masked gradient into a defined Tensor",
                        ),
                    ]
                    yield pytest.param(
                        dtype,
                        transposed,
                        mask,
                        shape,
                        marks=marks,
                        id=f"shape{shape_index}-{transposed}-{dtype}-{name}",
                    )


@pytest.mark.convolution_backward
@pytest.mark.parametrize("dtype,transposed,mask,shape", _cases(False))
def test_convolution_backward(dtype, transposed, mask, shape):
    bench = ConvolutionBackwardBenchmark(
        case_shape=shape,
        op_name="convolution_backward",
        torch_op=torch.ops.aten.convolution_backward.default,
        gems_op=flag_gems.convolution_backward,
        dtypes=[dtype],
        input_fn=partial(_inputs, transposed=transposed, mask=mask, out=False),
    )
    bench.run()


@pytest.mark.convolution_backward_out
@pytest.mark.parametrize("dtype,transposed,mask,shape", _cases(True))
def test_convolution_backward_out(dtype, transposed, mask, shape):
    bench = ConvolutionBackwardBenchmark(
        case_shape=shape,
        op_name="convolution_backward_out",
        torch_op=torch.ops.aten.convolution_backward.out,
        gems_op=flag_gems.convolution_backward_out,
        dtypes=[dtype],
        input_fn=partial(_inputs, transposed=transposed, mask=mask, out=True),
    )
    bench.run()
