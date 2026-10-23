import pytest
import torch

import flag_gems

from .base import Benchmark


class CastCharBenchmark(Benchmark):
    def get_input_iter(self, cur_dtype):
        for shape in [(1024,), (65536,), (1048576,), (16777216,)]:
            yield (torch.randn(shape, device=self.device, dtype=cur_dtype),)
        x = torch.randn((1024, 1024), device=self.device, dtype=cur_dtype)
        yield (x.t(),)
        yield (x[:, ::2],)


@pytest.mark.cast_Char
def test_cast_Char():
    bench = CastCharBenchmark(
        op_name="cast_Char",
        torch_op=torch.ops.aten._cast_Char.default,
        gems_op=flag_gems.cast_Char,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    )
    bench.run()
