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
        self.mock_code = ""
        self.arg_func_map = {
            binary_args: "binary_args",
            binary_int_args: "binary_int_args",
            ternary_args: "ternary_args",
            unary_arg: "unary_arg",
            unary_int_arg: "unary_int_arg",
            where_args: "where_args",
            cumsum_args: "cumsum_args",
            layer_norm_args: "layer_norm_args",
            cross_entropy_loss_args: "cross_entropy_loss_args",
            mv_args: "mv_args",
        }
        self.kwags_func_map = {
            flip_kwargs: "flip_kwargs",
        }

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
            torch.cuda.synchronize()
            start = time.time()
            for i in range(REPETITION):
                fn()
            torch.cuda.synchronize()
            end = time.time()
            latency = (end - start) / REPETITION * 1000
        else:
            latency = triton.testing.do_bench(
                fn,
                warmup=WARMUP,
                rep=REPETITION,
                return_mode="median",
                mock_code=self.mock_code,
                flag_gems_op_name=self.op_name,
            )
        # average latency in ms
        return latency

    def run(self):
        print(f"{self.op_name}")
        for size in self.sizes:
            print(f"{size}", end="")
            for dtype in self.dtypes:
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
                self.mock_code = f"""\
import torch
import flag_gems

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

def where_args(dtype, batch, size):
    inp1 = torch.randn([batch, size], dtype=dtype, device="cuda")
    inp2 = torch.randn([batch, size], dtype=dtype, device="cuda")
    condition = inp1 > 0
    return condition, inp1, inp2

def flip_kwargs(dtype, batch, size):
    return {{"dims": [0, 1]}}

def cumsum_args(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    return inp, 1

def layer_norm_args(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    weight = torch.randn(
        [
            size,
        ],
        dtype=dtype,
        device="cuda",
    )
    bias = torch.randn(
        [
            size,
        ],
        dtype=dtype,
        device="cuda",
    )
    return (
        inp,
        [
            size,
        ],
        weight,
        bias,
    )

def cross_entropy_loss_args(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    target = torch.randint(
        0,
        size,
        [
            batch,
        ],
        device="cuda",
    )
    return inp, target

def mv_args(dtype, batch, size):
    inp1 = torch.randn([size, size], dtype=dtype, device="cuda")
    inp2 = torch.randn([size], dtype=dtype, device="cuda")
    return inp1, inp2

class BenchmarkMock:
    def __init__(self):
        self.op = {self.torch_op.__module__}.\\
        {self.torch_op.__name__ if hasattr(self.torch_op, '__name__') else self.torch_op.__class__.__name__}
        self.arg_func = {self.arg_func_map.get(self.arg_func, "unknown")}
        self.kwargs_func = {self.kwags_func_map.get(self.kwargs_func, "None")}
        self.dtype = {dtype}
        self.batch = {self.batch}
        self.size = {size}

    def run(self):
        args = ()
        if self.arg_func is not None:
            args = self.arg_func(self.dtype, self.batch, self.size)
        kwargs = {{}}
        if self.kwargs_func is not None:
            kwargs = self.kwargs_func(self.dtype, self.batch, self.size)
        fn = lambda: self.op(*args, **kwargs)
        if 'use_gems' in globals():
            with flag_gems.use_gems():
                fn()
        else:
            fn()


def main():
    bm = BenchmarkMock()
    bm.run()


if __name__ == '__main__':
    main()
                """
                torch_perf = self.profile(self.torch_op, *args, **kwargs)
                if self.gems_op:
                    gems_perf = self.profile(self.gems_op, *args, **kwargs)
                else:
                    with flag_gems.use_gems():
                        self.mock_code = "use_gems=True\n" + self.mock_code
                        gems_perf = self.profile(self.torch_op, *args, **kwargs)
                print(f", {torch_perf}, {gems_perf}", end="")
            print()


FLOAT_DTYPES = [torch.float16, torch.float32]  # TODO: add torch.bfloat16
INT_DTYPES = [torch.int16, torch.int32]


DEFAULT_BATCH = 1
POINTWISE_BATCH = 1024
REDUCTION_BATCH = 1024
BLAS_BATCH = 16
SIZES = [i * 64 for i in range(1, 22, 5)]


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


def where_args(dtype, batch, size):
    inp1 = torch.randn([batch, size], dtype=dtype, device="cuda")
    inp2 = torch.randn([batch, size], dtype=dtype, device="cuda")
    condition = inp1 > 0
    return condition, inp1, inp2


def flip_kwargs(dtype, batch, size):
    return {"dims": [0, 1]}


def cumsum_args(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    return inp, 1


def layer_norm_args(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    weight = torch.randn(
        [
            size,
        ],
        dtype=dtype,
        device="cuda",
    )
    bias = torch.randn(
        [
            size,
        ],
        dtype=dtype,
        device="cuda",
    )
    return (
        inp,
        [
            size,
        ],
        weight,
        bias,
    )


def cross_entropy_loss_args(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    target = torch.randint(
        0,
        size,
        [
            batch,
        ],
        device="cuda",
    )
    return inp, target


def mv_args(dtype, batch, size):
    inp1 = torch.randn([size, size], dtype=dtype, device="cuda")
    inp2 = torch.randn([size], dtype=dtype, device="cuda")
    return inp1, inp2
