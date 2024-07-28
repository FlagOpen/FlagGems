import time

import torch
import triton

import flag_gems

from .conftest import CPU_MODE, DEVICE

WARMUP = 100
REPETITION = 1000
torch.backends.mlu.matmul.allow_tf32 = False


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
        self.gems_op = None
        self.arg_func = arg_func
        self.kwargs_func = kwargs_func
        self.dtypes = dtypes
        self.batch = batch
        self.sizes = sizes
        self.gems_op = None
        self.is_backward = is_backward

    def set_gems(self, gems_op):
        self.gems_op = gems_op

    def profile(self, op, *args, **kwargs):
        fn = lambda: op(*args, **kwargs)
        if self.is_backward:
            out = fn()
            dout = torch.randn_like(out)
            fn = lambda: out.backward(dout, retain_graph=True)
        if CPU_MODE:
            for i in range(WARMUP):
                fn()
            torch.mlu.synchronize()
            start = time.time()
            for i in range(REPETITION):
                fn()
            torch.mlu.synchronize()
            end = time.time()
            latency = (end - start) / REPETITION * 1000
        else:
            latency = triton.testing.do_bench(
                fn,
                warmup=WARMUP,
                rep=REPETITION,
                return_mode="median",
            )
        # average latency in ms
        return latency

    def run(self):
        for size in self.sizes:
            kep = []
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
                # print(f"\t{speedup}", end="")
                kep.append(speedup)
            print(
                f"\nOperator_Speedup_Test_Result("
                + ":".join([str(x) for x in self.dtypes])
                + f"):\t{self.op_name}\t{str(size)}\t"
                + "\t".join([str(x) for x in kep])
            )

    def run(self):
        kep = []
        for dtype in self.dtypes:
            # print(f"\nOperator {self.op_name} Speedup Test ({dtype})")
            speedup = 0
            for size in self.sizes:
                args = ()
                if self.arg_func is not None:
                    args = self.arg_func(dtype, self.batch, size)
                if self.is_backward:
                    args = tuple(
                        a.clone().requires_grad_() if torch.is_tensor(a) and torch.is_floating_point(a)
                        else a
                        for a in args
                    )

                kwargs = {}
                if self.kwargs_func is not None:
                    kwargs = self.kwargs_func(dtype, self.batch, size)

                torch_perf = self.profile(self.torch_op, *args, **kwargs)
                if self.gems_op:
                    gems_perf = self.profile(self.gems_op, *args, **kwargs)
                else:
                    with flag_gems.use_gems():
                        gems_perf = self.profile(self.torch_op, *args, **kwargs)
                spd = torch_perf / gems_perf
                speedup += spd
            speedup /= len(self.sizes)
            kep.append(speedup)
        print(f"\nOperator_Speedup_Test_Result(" + ":".join(
            [str(x) for x in self.dtypes]) + f"):\t{self.op_name}\t" + "\t".join([str(x) for x in kep]))


FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
BATCH = 1024
SIZES = [32, 96, 8192, 20480, 32768]


def unary_arg(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device=DEVICE)
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
        inp = torch.randn([batch, size], dtype=dtype, device=DEVICE)
        weight = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device=DEVICE,
        )
        bias = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device=DEVICE,
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