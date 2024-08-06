import time

import torch
import triton

import flag_gems

from .conftest import CPU_MODE

WARMUP = 100
REPETITION = 1000
# torch.backends.cuda.matmul.allow_tf32 = False


class Benchmark:
    def __init__(
        self, op_name, torch_op, arg_func, dtypes, batch, sizes, kwargs_func=None
    ):
        self.op_name = op_name
        self.torch_op = torch_op
        self.arg_func = arg_func
        self.kwargs_func = kwargs_func
        self.dtypes = dtypes
        self.batch = batch
        self.sizes = sizes
        self.gems_op = None

    def set_gems(self, gems_op):
        self.gems_op = gems_op

    def profile(self, op, *args, **kwargs):
        if CPU_MODE:
            for i in range(WARMUP):
                op(*args, **kwargs)
            torch.musa.synchronize()
            start = time.time()
            for i in range(REPETITION):
                op(*args, **kwargs)
            torch.musa.synchronize()
            end = time.time()
            latency = (end - start) / REPETITION * 1000
        else:
            latency = triton.musa_testing.do_bench(
                lambda: op(*args, **kwargs),
                warmup=WARMUP,
                rep=REPETITION,
                return_mode="median",
            )
        # average latency in ms
        return latency

    def run(self):
        for size in self.sizes:
            print(f"{BATCH}x{size}", end="")
            for dtype in self.dtypes:
                args = ()
                if self.arg_func is not None:
                    args = self.arg_func(dtype, self.batch, size)

                kwargs = {}
                if self.kwargs_func is not None:
                    kwargs = self.kwargs_func(dtype, self.batch, size)

                torch_perf = self.profile(self.torch_op, *args, **kwargs)
                if self.gems_op:
                    gems_perf = self.profile(self.gems_op, *args, **kwargs)
                else:
                    with flag_gems.use_gems():
                        gems_perf = self.profile(self.torch_op, *args, **kwargs)
                speedup = torch_perf / gems_perf
                print(f", {speedup}", end="")
            print()


# FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
FLOAT_DTYPES = [torch.float16, torch.float32]
BATCH = 1024
SIZES = [32, 96, 8192, 20480, 32768]


def unary_arg(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="musa")
    return (inp,)


def test_perf_gelu():
    bench = Benchmark(
        op_name="gelu",
        torch_op=torch.nn.functional.gelu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_sigmoid():
    bench = Benchmark(
        op_name="sigmoid",
        torch_op=torch.sigmoid,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_layernorm():
    def layer_norm_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device="musa")
        weight = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device="musa",
        )
        bias = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device="musa",
        )
        return (
            inp,
            [
                size,
            ],
            weight,
            bias,
        )

    bench = Benchmark(
        op_name="layernorm",
        torch_op=torch.layer_norm,
        arg_func=layer_norm_args,
        dtypes=FLOAT_DTYPES,
        batch=BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_softmax():
    bench = Benchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=BATCH,
        sizes=SIZES,
    )
    bench.run()

