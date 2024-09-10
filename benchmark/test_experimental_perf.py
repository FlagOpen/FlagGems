import torch

import flag_gems.experimental

from .performance_utils import Benchmark


def test_perf_rad2_fft():
    def rad2_fft_kwargs(dtype, batch, size):
        x = torch.randn((size,), device="cuda", dtype=dtype)
        return (x,)

    gems_op = flag_gems.experimental.fft.rad2_fft

    bench = Benchmark(
        op_name="fft.fft",
        torch_op=torch.fft.fft,
        arg_func=rad2_fft_kwargs,
        dtypes=[torch.cfloat],
        batch=[1],
        sizes=[i * 1024 for i in range(1, 100, 10)],
    )
    bench.set_gems(gems_op)
    bench.run()
