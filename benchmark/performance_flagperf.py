import time

import torch
import triton

import flag_gems
import pytest

from .conftest import CPU_MODE

WARMUP = 50
REPETITION = 100
torch.backends.mlu.matmul.allow_tf32 = False


FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32]

M_ELEMENTS = [16, 1024]
SIZE = 1024


class Benchmark:
    def __init__(
        self,
        op_name,
        torch_op,
        arg_func,
        dtypes,
        m_elements,
        size,
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
        self.m_elements = m_elements
        self.size = size
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
        for m_elements in self.m_elements:
            kep = []
            for dtype in self.dtypes:
                args = ()
                if self.arg_func is not None:
                    args = self.arg_func(dtype, m_elements, self.size)

                if self.is_backward:
                    args = tuple(
                        (
                            a.clone().requires_grad_()
                            if torch.is_tensor(a) and torch.is_floating_point(a)
                            else a
                        )
                        for a in args
                    )

                kwargs = {}
                if self.kwargs_func is not None:
                    kwargs = self.kwargs_func(dtype, m_elements, self.size)

                torch_perf = self.profile(self.torch_op, *args, **kwargs)
                if self.gems_op:
                    gems_perf = self.profile(self.gems_op, *args, **kwargs)
                else:
                    with flag_gems.use_gems():
                        gems_perf = self.profile(self.torch_op, *args, **kwargs)
                speedup = torch_perf / gems_perf

                kep.append(speedup)
            print(
                f"\nOperator_Speedup_Test_Result("
                + ":".join([str(x) for x in self.dtypes])
                + f"):\t{self.op_name}\t\t{m_elements}\t"
                + "\t".join([str(x) for x in kep])
            )


def unary_arg(dtype, m_elements, size):
    inp = torch.randn([m_elements, size, size], dtype=dtype, device="cuda")
    return (inp,)


def unary_int_arg(dtype, m_elements, size):
    inp = torch.randint(
        low=0, high=0x7FFF, size=[m_elements, size, size], dtype=dtype, device="cpu"
    ).to("cuda")
    return (inp,)


def binary_args(dtype, m_elements, size):
    inp1 = torch.randn([m_elements, size, size], dtype=dtype, device="cuda")
    inp2 = torch.randn([m_elements, size, size], dtype=dtype, device="cuda")
    return inp1, inp2


def binary_int_args(dtype, m_elements, size):
    inp1 = torch.randint(
        low=0, high=0x7FFF, size=[m_elements, size, size], dtype=dtype, device="cpu"
    ).to("cuda")
    inp2 = torch.randint(
        low=0, high=0x7FFF, size=[m_elements, size, size], dtype=dtype, device="cpu"
    ).to("cuda")
    return inp1, inp2


def test_perf_abs():
    bench = Benchmark(
        op_name="abs",
        torch_op=torch.abs,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_add():
    bench = Benchmark(
        op_name="add",
        torch_op=torch.add,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()

def test_perf_triu():
    def triu_args(dtype, m_elements, size):
        inp = torch.randn([m_elements * 50, size * 50], dtype=dtype, device=DEVICE)
        return (inp, )
    bench = Benchmark(
        op_name="triu",
        torch_op=torch.triu,
        arg_func=triu_args,
        dtypes=FLOAT_DTYPES,
        m_elements=[1024],
        size=SIZE,
    )
    bench.run()


def test_perf_bitwiseand_int():
    bench = Benchmark(
        op_name="bitwiseand_int",
        torch_op=torch.bitwise_and,
        arg_func=binary_int_args,
        dtypes=INT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_bitwisenot_int():
    bench = Benchmark(
        op_name="bitwisenot_int",
        torch_op=torch.bitwise_not,
        arg_func=unary_int_arg,
        dtypes=INT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_bitwiseor_int():
    bench = Benchmark(
        op_name="bitwiseor_int",
        torch_op=torch.bitwise_or,
        arg_func=binary_int_args,
        dtypes=INT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_cos():
    bench = Benchmark(
        op_name="cos",
        torch_op=torch.cos,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_div():
    bench = Benchmark(
        op_name="div",
        torch_op=torch.div,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_eq():
    bench = Benchmark(
        op_name="eq",
        torch_op=torch.eq,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_exp():
    bench = Benchmark(
        op_name="exp",
        torch_op=torch.exp,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_ge():
    bench = Benchmark(
        op_name="ge",
        torch_op=torch.ge,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_gelu():
    bench = Benchmark(
        op_name="gelu",
        torch_op=torch.nn.functional.gelu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_gt():
    bench = Benchmark(
        op_name="gt",
        torch_op=torch.gt,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_isinf():
    bench = Benchmark(
        op_name="isinf",
        torch_op=torch.isinf,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_isnan():
    bench = Benchmark(
        op_name="isnan",
        torch_op=torch.isnan,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_le():
    bench = Benchmark(
        op_name="le",
        torch_op=torch.le,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_lt():
    bench = Benchmark(
        op_name="lt",
        torch_op=torch.lt,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_mul():
    bench = Benchmark(
        op_name="mul",
        torch_op=torch.mul,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_ne():
    bench = Benchmark(
        op_name="ne",
        torch_op=torch.ne,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_neg():
    bench = Benchmark(
        op_name="neg",
        torch_op=torch.neg,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_pow():
    bench = Benchmark(
        op_name="pow",
        torch_op=torch.pow,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_reciprocal():
    bench = Benchmark(
        op_name="reciprocal",
        torch_op=torch.reciprocal,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_relu():
    bench = Benchmark(
        op_name="relu",
        torch_op=torch.nn.functional.relu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_rsqrt():
    bench = Benchmark(
        op_name="rsqrt",
        torch_op=torch.rsqrt,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_sigmoid():
    bench = Benchmark(
        op_name="sigmoid",
        torch_op=torch.sigmoid,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_sigmoid_backward():
    bench = Benchmark(
        op_name="sigmoid",
        torch_op=torch.sigmoid,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
        is_backward=True,
    )
    bench.run()


def test_perf_silu():
    bench = Benchmark(
        op_name="silu",
        torch_op=torch.nn.functional.silu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_sin():
    bench = Benchmark(
        op_name="sin",
        torch_op=torch.sin,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_sub():
    bench = Benchmark(
        op_name="sub",
        torch_op=torch.sub,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()

def test_perf_rsub():
    bench = Benchmark(
        op_name="rsub",
        torch_op=torch.sub,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()

def test_perf_dropout_backward():
    bench = Benchmark(
        op_name="dropout",
        torch_op=torch.tanh,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
        is_backward=True,
    )
    bench.run()


def test_perf_dropout():
    bench = Benchmark(
        op_name="tanh",
        torch_op=torch.tanh,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_tanh_backward():
    bench = Benchmark(
        op_name="tanh",
        torch_op=torch.tanh,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
        is_backward=True,
    )
    bench.run()


def test_perf_tanh():
    bench = Benchmark(
        op_name="tanh",
        torch_op=torch.tanh,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_all():
    def all_arg(dtype, m_elements, size):
        inp = torch.arange(0, m_elements * size * size, dtype=dtype, device="cuda")
        return (inp,)
    bench = Benchmark(
        op_name="all",
        torch_op=torch.all,
        arg_func=all_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_cross_entropy_loss():
    def cross_entropy_loss_args(dtype, m_elements, size):
        inp = torch.randn([m_elements, size], dtype=dtype, device="cuda")
        target = torch.randint(
            0,
            size,
            [
                m_elements,
            ],
            device="cuda",
        )
        return inp, target

    bench = Benchmark(
        op_name="cross_entropy_loss",
        torch_op=torch.nn.CrossEntropyLoss(),
        arg_func=cross_entropy_loss_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=128000,
    )
    bench.run()


def test_perf_cross_entropy_loss_backward():
    def cross_entropy_loss_args(dtype, m_elements, size):
        inp = torch.randn([m_elements, size], dtype=dtype, device="cuda")
        target = torch.randint(
            0,
            size,
            [
                m_elements,
            ],
            device="cuda",
        )
        return inp, target

    bench = Benchmark(
        op_name="cross_entropy_loss",
        torch_op=torch.nn.CrossEntropyLoss(),
        arg_func=cross_entropy_loss_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=128000,
        is_backward=True,
    )
    bench.run()


def test_perf_log_softmax():
    def log_softmax_arg(dtype, m_elements, size):
        inp = torch.randn([m_elements * 1024, size], dtype=dtype, device="cuda")
        return (inp,)
    bench = Benchmark(
        op_name="log_softmax",
        torch_op=torch.nn.functional.log_softmax,
        arg_func=log_softmax_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=640,
    )
    bench.run()


def test_perf_log_softmax_backward():
    def log_softmax_arg(dtype, m_elements, size):
        inp = torch.randn([m_elements * 1024, size], dtype=dtype, device="cuda")
        return (inp,)
    bench = Benchmark(
        op_name="log_softmax",
        torch_op=torch.nn.functional.log_softmax,
        arg_func=log_softmax_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=640,
        is_backward=True,
    )
    bench.run()


def test_perf_max():
    bench = Benchmark(
        op_name="max",
        torch_op=torch.max,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()

def test_perf_amax():
    pytest.skip("exceeds triton maximum tensor numel failed")
    def amax_arg(dtype, m_elements, size):
        inp = torch.randn([m_elements * 80, size * 80], dtype=dtype, device=DEVICE)
        return (inp, 1)
    bench = Benchmark(
        op_name="amax",
        torch_op=torch.amax,
        arg_func=amax_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()

def test_perf_argmax():
    pytest.skip("grid is greater than max_grid_size")
    def argmax_arg(dtype, m_elements, size):
        inp = torch.randn([m_elements * 80, size * 80], dtype=dtype, device=DEVICE)
        return (inp, 1)
    bench = Benchmark(
        op_name="argmax",
        torch_op=torch.argmax,
        arg_func=argmax_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_mean():
    bench = Benchmark(
        op_name="mean",
        torch_op=torch.mean,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_min():
    bench = Benchmark(
        op_name="min",
        torch_op=torch.min,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_prod():
    bench = Benchmark(
        op_name="prod",
        torch_op=torch.prod,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_softmax():
    def softmax_arg(dtype, m_elements, size):
        inp = torch.randn([m_elements, size, size], dtype=dtype, device="cuda")
        return inp, 1
    bench = Benchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        arg_func=softmax_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_softmax_backward():
    def softmax_arg(dtype, m_elements, size):
        inp = torch.randn([m_elements, size, size], dtype=dtype, device="cuda")
        return inp, 1
    bench = Benchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        arg_func=softmax_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
        is_backward=True,
    )
    bench.run()


def test_perf_sum():
    bench = Benchmark(
        op_name="sum",
        torch_op=torch.sum,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()

def test_perf_outer():
    def outer_args(dtype, m_elements, size):
        inp1 = torch.randn([m_elements * 10], dtype=dtype, device=DEVICE)
        inp2 = torch.randn([size * 10], dtype=dtype, device=DEVICE)
        return inp1, inp2
    bench = Benchmark(
        op_name="outer",
        torch_op=torch.outer,
        arg_func=outer_args,
        dtypes=FLOAT_DTYPES,
        m_elements=[1024],
        size=SIZE,
    )
    bench.run()

def test_perf_bmm():
    def bmm_args(dtype, m_elements, size):
        inp1 = torch.randn([m_elements, size, size], dtype=dtype, device="cuda")
        inp2 = torch.randn([m_elements, size, size], dtype=dtype, device="cuda")
        return inp1, inp2

    bench = Benchmark(
        op_name="bmm",
        torch_op=torch.bmm,
        arg_func=bmm_args,
        dtypes=FLOAT_DTYPES,
        m_elements=[2],
        size=4096,
    )
    bench.run()


def test_perf_mm():
    def mm_args(dtype, m_elements, size):
        inp1 = torch.randn([size, size], dtype=dtype, device="cuda")
        inp2 = torch.randn([size, size], dtype=dtype, device="cuda")
        return inp1, inp2

    bench = Benchmark(
        op_name="mm",
        torch_op=torch.mm,
        arg_func=mm_args,
        dtypes=FLOAT_DTYPES,
        m_elements=[8192],
        size=8192,
    )
    bench.run()

def test_perf_addmm():
    def addmm_args(dtype, m_elements, size):
        inp1 = torch.randn([size], dtype=dtype, device=DEVICE)
        inp2 = torch.randn([size, size], dtype=dtype, device=DEVICE)
        inp3 = torch.randn([size, size], dtype=dtype, device=DEVICE)
        return inp1, inp2, inp3

    bench = Benchmark(
        op_name="addmm",
        torch_op=torch.addmm,
        arg_func=addmm_args,
        dtypes=FLOAT_DTYPES,
        m_elements=[4096],
        size=4096,
    )
    bench.run()


def test_perf_mv():
    def mv_args(dtype, m_elements, size):
        inp1 = torch.randn([m_elements, size], dtype=dtype, device="cuda")
        inp2 = torch.randn([size], dtype=dtype, device="cuda")
        return inp1, inp2

    bench = Benchmark(
        op_name="mv",
        torch_op=torch.mv,
        arg_func=mv_args,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()

def test_perf_var_mean():
    bench = Benchmark(
        op_name="var_mean",
        torch_op=torch.var_mean,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()


def test_perf_vector_norm():
    bench = Benchmark(
        op_name="vector_norm",
        torch_op=torch.linalg.vector_norm,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        m_elements=M_ELEMENTS,
        size=SIZE,
    )
    bench.run()

def test_perf_groupnorm():
    pytest.skip("Nram exceed failed")
    def group_norm_args(dtype, batch, size):
        C = 6
        G = 3
        inp = torch.randn([batch, C, size], dtype=dtype, device="cuda")
        weight = torch.randn(
            [
                C,
            ],
            dtype=dtype,
            device="cuda",
        )
        bias = torch.randn(
            [
                C,
            ],
            dtype=dtype,
            device="cuda",
        )
        return inp, G, weight, bias

    bench = Benchmark(
        op_name="groupnorm",
        torch_op=torch.nn.functional.group_norm,
        arg_func=group_norm_args,
        dtypes=FLOAT_DTYPES,
        m_elements=[20],
        size=65536,
    )
    bench.run()

def test_perf_groupnorm_backward():
    pytest.skip("Nram exceed failed")
    def group_norm_args(dtype, batch, size):
        C = 6
        G = 3
        inp = torch.randn([batch, C, size], dtype=dtype, device="cuda")
        weight = torch.randn(
            [
                C,
            ],
            dtype=dtype,
            device="cuda",
        )
        bias = torch.randn(
            [
                C,
            ],
            dtype=dtype,
            device="cuda",
        )
        return inp, G, weight, bias

    bench = Benchmark(
        op_name="groupnorm",
        torch_op=torch.nn.functional.group_norm,
        arg_func=group_norm_args,
        dtypes=FLOAT_DTYPES,
        m_elements=[20],
        size=65536,
        is_backward=True,
    )
    bench.run()

def test_perf_layernorm():
    def layer_norm_args(dtype, batch, size):
        C = 6
        inp = torch.randn([batch, C, size], dtype=dtype, device="cuda")
        weight = torch.randn(
            [
                C, size
            ],
            dtype=dtype,
            device="cuda",
        )
        bias = torch.randn(
            [
                C, size
            ],
            dtype=dtype,
            device="cuda",
        )
        return (
            inp,
            [
                C, size
            ],
            weight,
            bias,
        )

    bench = Benchmark(
        op_name="layernorm",
        torch_op=torch.layer_norm,
        arg_func=layer_norm_args,
        dtypes=FLOAT_DTYPES,
        m_elements=[20],
        size=1048576,
    )
    bench.run()

