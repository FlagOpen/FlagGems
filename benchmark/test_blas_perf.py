import torch
import pytest
from benchmark.op_configs import op_configs

from .performance_utils import Benchmark, device, DEFAULT_METRICS,get_shape

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

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "addmm"])
@pytest.mark.addmm
def test_perf_addmm(config):
    def addmm_args(dtype, batch, size):
        shape=get_shape(batch, size)
        bias = torch.randn(shape, dtype=dtype, device=device)
        inp1 = torch.randn(shape, dtype=dtype, device=device)
        inp2 = torch.randn(shape, dtype=dtype, device=device)
        return bias, inp1, inp2

    bench = BlasBenchmark(
        op_name="addmm",
        torch_op=torch.addmm,
        arg_func=addmm_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "bmm"])
@pytest.mark.bmm
def test_perf_bmm(config):
    def bmm_args(dtype, batch, size):
        shape=get_shape(batch, size)
        inp1 = torch.randn(shape, dtype=dtype, device=device)
        inp2 = torch.randn(shape, dtype=dtype, device=device)
        return inp1, inp2

    bench = BlasBenchmark(
        op_name="bmm",
        torch_op=torch.bmm,
        arg_func=bmm_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()


@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "mm"])
@pytest.mark.mm
def test_perf_mm(config):
    def mm_args(dtype, batch, size):
        shape=get_shape(batch, size)
        inp1 = torch.randn(shape, dtype=dtype, device=device)
        inp2 = torch.randn(shape, dtype=dtype, device=device)
        return inp1, inp2

    bench = BlasBenchmark(
        op_name="mm",
        torch_op=torch.mm,
        arg_func=mm_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "mv"])
@pytest.mark.mv
def test_perf_mv(config):
    def mv_args(dtype, batch, size):
        shape=get_shape(batch, size)
        inp1 = torch.randn(shape, dtype=dtype, device=device)
        inp2 = torch.randn([shape[0]], dtype=dtype, device=device)
        return inp1, inp2

    bench = BlasBenchmark(
        op_name="mv",
        torch_op=torch.mv,
        arg_func=mv_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "outer"])
@pytest.mark.outer
def test_perf_outer(config):
    def outer_args(dtype, batch, size):
        shape=get_shape(batch, size)
        inp1 = torch.randn(shape, dtype=dtype, device=device)
        inp2 = torch.randn(shape, dtype=dtype, device=device)
        return inp1, inp2

    bench = Benchmark(
        op_name="outer",
        torch_op=torch.outer,
        arg_func=outer_args,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()
