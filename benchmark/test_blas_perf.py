import torch
import pytest

from .performance_utils import BLAS_BATCH, DEFAULT_BATCH, FLOAT_DTYPES, SIZES, Benchmark, device,DEFAULT_METRICS

class BlasBenchmark(Benchmark):
    """
    benchmark for blas
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def get_tflops(self, op, *args, **kwargs):
        """This method is currently not really implemented and serves as a placeholder.
        A proper implementation will be developed in the future."""
        total_flops = 0
        # shape(m,k)(k,n)
        # total_flops mxnx2k
        if self.op_name == "mm":
            total_flops = args[0].shape[0] * args[0].shape[1] * args[1].shape[1] * 2
        # shape(m,n)(n,p)
        # total_flops mxpx(2n+1)
        if self.op_name == "addmm":
            total_flops = (
                args[0].shape[0] * args[1].shape[1] * (args[1].shape[0] * 2 + 1)
            )
        # shape(b,n,m), (b,m,p)
        # total_flops bxnxpx2m
        if self.op_name == "bmm":
            total_flops = (
                args[0].shape[0]
                * args[0].shape[1]
                * args[1].shape[2]
                * 2
                * args[0].shape[2]
            )
        # shape(n,m)(m,)
        # total_flops n*2m
        if self.op_name == "mv":
            total_flops = args[0].shape[0] * 2 * args[0].shape[1]

        return total_flops


@pytest.mark.addmm
def test_perf_addmm():
    def addmm_args(dtype, batch, size):
        bias = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device=device,
        )
        inp1 = torch.randn([size, size], dtype=dtype, device=device)
        inp2 = torch.randn([size, size], dtype=dtype, device=device)
        return bias, inp1, inp2

    bench = BlasBenchmark(
        op_name="addmm",
        torch_op=torch.addmm,
        arg_func=addmm_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.bmm
def test_perf_bmm():
    def bmm_args(dtype, batch, size):
        inp1 = torch.randn([batch, size, size], dtype=dtype, device=device)
        inp2 = torch.randn([batch, size, size], dtype=dtype, device=device)
        return inp1, inp2

    bench = BlasBenchmark(
        op_name="bmm",
        torch_op=torch.bmm,
        arg_func=bmm_args,
        dtypes=FLOAT_DTYPES,
        batch=BLAS_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.mm
def test_perf_mm():
    def mm_args(dtype, batch, size):
        inp1 = torch.randn([size, size], dtype=dtype, device=device)
        inp2 = torch.randn([size, size], dtype=dtype, device=device)
        return inp1, inp2

    bench = BlasBenchmark(
        op_name="mm",
        torch_op=torch.mm,
        arg_func=mm_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=SIZES,
    )
    bench.run()


@pytest.mark.mv
def test_perf_mv():
    def mv_args(dtype, batch, size):
        inp1 = torch.randn([size, size], dtype=dtype, device=device)
        inp2 = torch.randn([size], dtype=dtype, device=device)
        return inp1, inp2

    bench = BlasBenchmark(
        op_name="mv",
        torch_op=torch.mv,
        arg_func=mv_args,
        dtypes=FLOAT_DTYPES,
        batch=BLAS_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_outer():
    def outer_args(dtype, batch, size):
        inp1 = torch.randn([size], dtype=dtype, device=device)
        inp2 = torch.randn([size], dtype=dtype, device=device)
        return inp1, inp2

    bench = Benchmark(
        op_name="outer",
        torch_op=torch.outer,
        arg_func=outer_args,
        dtypes=FLOAT_DTYPES,
        batch=DEFAULT_BATCH,
        sizes=SIZES,
    )
    bench.run()
