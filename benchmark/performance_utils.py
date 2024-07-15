import time

import torch
import triton

import flag_gems

from .conftest import CPU_MODE

WARMUP = 100
REPETITION = 1000
torch.backends.cuda.matmul.allow_tf32 = False


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
            torch.cuda.synchronize()
            start = time.time()
            for i in range(REPETITION):
                op(*args, **kwargs)
            torch.cuda.synchronize()
            end = time.time()
            latency = (end - start) / REPETITION * 1000
        else:
            latency = triton.testing.do_bench(
                lambda: op(*args, **kwargs),
                warmup=WARMUP,
                rep=REPETITION,
                return_mode="median",
            )
        # average latency in ms
        return latency

    def run(self):
        for dtype in self.dtypes:
            print(f"Operator {self.op_name} Performance Test ({dtype})")
            print("Size        Torch Latency (ms)   Gems Latency (ms)")
            print("--------------------------------------------------")
            for size in self.sizes:
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
                print(f"{size: <10}{torch_perf: >20.6}{gems_perf: >20.6}")


FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32]


DEFAULT_BATCH = 1
POINTWISE_BATCH = 1024
REDUCTION_BATCH = 1024
BLAS_BATCH = 16
SIZES = [i * 1024 for i in range(1, 81, 5)]


def unary_arg(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    return (inp,)


def unary_int_arg(dtype, batch, size):
    inp = torch.randint(
        low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device="cuda"
    )
    return (inp,)


def binary_args(dtype, batch, size):
    inp1 = torch.randn([batch, size], dtype=dtype, device="cuda")
    inp2 = torch.randn([batch, size], dtype=dtype, device="cuda")
    return inp1, inp2


def binary_int_args(dtype, batch, size):
    inp1 = torch.randint(
        low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device="cuda"
    )
    inp2 = torch.randint(
        low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device="cuda"
    )
    return inp1, inp2


def ternary_args(dtype, batch, size):
    inp1 = torch.randn([batch, size], dtype=dtype, device="cuda")
    inp2 = torch.randn([batch, size], dtype=dtype, device="cuda")
    inp3 = torch.randn([batch, size], dtype=dtype, device="cuda")
    return inp1, inp2, inp3
