import time

import torch
import triton

import flag_gems

from .attri_util import BenchmarkResult, BenckmarkMatrics
from .conftest import Config

torch.backends.cuda.matmul.allow_tf32 = False


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
        self.results = []

    def set_gems(self, gems_op):
        self.gems_op = gems_op

    def get_latency(self, op, *args, **kwargs):
        fn = lambda: op(*args, **kwargs)
        if self.is_backward:
            out = fn()
            dout = torch.randn_like(out)
            fn = lambda: out.backward(dout, retain_graph=True)
        if Config.cpu_mode:
            for i in range(Config.warm_up):
                fn()
            torch.cuda.synchronize()
            start = time.time()
            for i in range(Config.repetition):
                fn()
            torch.cuda.synchronize()
            end = time.time()
            latency = (end - start) / Config.repetition * 1000
        else:
            latency = triton.testing.do_bench(
                fn,
                warmup=Config.warm_up,
                rep=Config.repetition,
                return_mode="median",
            )
        # average latency in ms
        return latency

    def get_tflops(self, op, *args, **kwargs):
        """not implemented"""
        from torch.utils.flop_counter import FlopCounterMode

        fn = lambda: op(*args, **kwargs)
        with FlopCounterMode(display=False) as flop_counter:
            fn()
        return flop_counter.get_total_flops()

    def run(self):
        mode_str = "cpu" if Config.cpu_mode else "cuda"
        # print("")
        for dtype in self.dtypes:
            # print(
            #     f"Operator {self.op_name} Performance Test (dtype={dtype}, mode={mode_str})"
            # )
            # print("Size      Torch Latency (ms)    Gems Latency (ms)    Gems Speedup    Size Detail")
            # print("--------------------------------------------------------------------------------")
            matrics = []
            for size in self.sizes:
                args = ()
                if self.arg_func is not None:
                    args = self.arg_func(dtype, self.batch, size)
                if self.is_backward:
                    args = tuple(
                        a.clone().requires_grad_()
                        if torch.is_tensor(a) and torch.is_floating_point(a)
                        else a
                        for a in args
                    )

                kwargs = {}
                if self.kwargs_func is not None:
                    kwargs = self.kwargs_func(dtype, self.batch, size)

                torch_latency = self.get_latency(self.torch_op, *args, **kwargs)
                if self.gems_op:
                    gems_latency = self.get_latency(self.gems_op, *args, **kwargs)
                else:
                    with flag_gems.use_gems():
                        gems_latency = self.get_latency(self.torch_op, *args, **kwargs)
                speedup = torch_latency / gems_latency

                size_product = 1
                [size_product := size_product * num for num in size]

                # tflops is decided by the operation suanfa. so we no need to fenbie cal torch tflops or gems tflops.
                tflops = self.get_tflops(self.torch_op, *args, **kwargs)
                utilization = tflops / gems_latency / 1e12 * 1e3
                # print(
                #     f"{size_product: <10}{torch_latency: >18.6}{gems_latency: >21.6}{speedup: >16.3}{' ' * 5}{size}"
                # )
                matric = BenckmarkMatrics(
                    shape=size_product,
                    shape_detail=size,
                    latency_base=torch_latency,
                    latency=gems_latency,
                    speedup=speedup,
                    tflops=tflops,
                    utilization=utilization,
                )
                matrics.append(matric)
            result = BenchmarkResult(
                op_name=self.op_name, dtype=str(dtype), mode=mode_str, result=matrics
            )
            print(result)


FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32]


DEFAULT_BATCH = 1
POINTWISE_BATCH = 1024
REDUCTION_BATCH = 1024
BLAS_BATCH = 16
SIZES = [i * 64 for i in range(1, 22, 5)]


def unary_arg_old(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    return (inp,)


def unary_arg(dtype, batch, shape):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cuda")
    elif dtype in INT_DTYPES:
        inp = torch.randint(low=0, high=0x7FFF, size=shape, dtype=dtype, device="cuda")
    return (inp,)


def binary_args_old(dtype, batch, size):
    if dtype in FLOAT_DTYPES:
        inp1 = torch.randn([batch, size], dtype=dtype, device="cuda")
        inp2 = torch.randn([batch, size], dtype=dtype, device="cuda")
    elif dtype in INT_DTYPES:
        inp1 = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            [batch, size],
            dtype=dtype,
            device="cuda",
        )
        inp2 = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            [batch, size],
            dtype=dtype,
            device="cuda",
        )
    return inp1, inp2


def binary_args(dtype, batch, shape):
    if dtype in FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device="cuda")
        inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    elif dtype in INT_DTYPES:
        inp1 = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cuda",
        )
        inp2 = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cuda",
        )
    return inp1, inp2


def ternary_args(dtype, batch, shape):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    inp3 = torch.randn(shape, dtype=dtype, device="cuda")
    return inp1, inp2, inp3
