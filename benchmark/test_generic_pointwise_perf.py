import torch
import pytest
from benchmark.op_configs import op_configs

from .performance_utils import (
    Benchmark,
    unary_arg,
    device,
)

@pytest.mark.parametrize("config", [c for c in op_configs if c["op_name"] == "triu"])
@pytest.mark.triu
def test_perf_triu(config):
    bench = Benchmark(
        op_name="triu",
        torch_op=torch.triu,
        arg_func=unary_arg,
        **{k: v for k, v in config.items() if k in ["dtypes", "batch", "sizes"]},
    )
    bench.run()

