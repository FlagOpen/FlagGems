import time

import torch
import triton

import flag_gems
from .conftest import CPU_MODE, DEVICE

WARMUP = 100
REPETITION = 1000


class Benchmark:
    def __init__(self, op_name, torch_op, arg_func, dtype, batch, sizes):
        self.op_name = op_name
        self.torch_op = torch_op
        self.gems_op = None
        self.arg_func = arg_func
        self.dtype = dtype
        self.batch = batch
        self.sizes = sizes
        self.gems_op = None

    def set_gems(self, gems_op):
        self.gems_op = gems_op

    def profile(self, op, *args):
        if CPU_MODE:
            for i in range(WARMUP):
                op(*args)
            torch.mlu.synchronize()
            start = time.time()
            for i in range(REPETITION):
                op(*args)
            torch.mlu.synchronize()
            end = time.time()
            latency = (end - start) / REPETITION * 1000
        else:
            latency = triton.testing.do_bench(
                lambda: op(*args), warmup=WARMUP, rep=REPETITION, return_mode="median"
            )
        # average latency in ms
        return latency

    def run(self):
        # print(f"\nOperator {self.op_name} Speedup Test ({self.dtype})")
        speedup = 0
        for size in self.sizes:
            args = self.arg_func(self.dtype, self.batch, size)
            torch_perf = self.profile(self.torch_op, *args)
            if self.gems_op is None:
                with flag_gems.use_gems():
                    gems_perf = self.profile(self.torch_op, *args)
            else:
                gems_perf = self.profile(self.gems_op, *args)
            spd = torch_perf / gems_perf
            speedup += spd
            # print(f"{size: <10}{torch_perf: >20.6}{gems_perf: >20.6}")
        speedup /= len(self.sizes)
        print(f"\nOperator_Speedup_Test_Result:\t{self.op_name}\t{self.dtype}\t{speedup}")


FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32]


DEFAULT_BATCH = 1
POINTWISE_BATCH = 1024
REDUCTION_BATCH = 1024
BLAS_BATCH = 16
SIZES = [i * 64 for i in range(1, 22, 5)]


def unary_arg(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device=DEVICE)
    return (inp,)


def unary_int_arg(dtype, batch, size):
    inp = torch.randint(
        low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device="cpu"
    ).to(DEVICE)
    return (inp,)


def binary_args(dtype, batch, size):
    inp1 = torch.randn([batch, size], dtype=dtype, device=DEVICE)
    inp2 = torch.randn([batch, size], dtype=dtype, device=DEVICE)
    return inp1, inp2


def binary_int_args(dtype, batch, size):
    inp1 = torch.randint(
        low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device="cpu"
    ).to(DEVICE)
    inp2 = torch.randint(
        low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device="cpu"
    ).to(DEVICE)
    return inp1, inp2


def ternary_args(dtype, batch, size):
    inp1 = torch.randn([batch, size], dtype=dtype, device=DEVICE)
    inp2 = torch.randn([batch, size], dtype=dtype, device=DEVICE)
    inp3 = torch.randn([batch, size], dtype=dtype, device=DEVICE)
    return inp1, inp2, inp3