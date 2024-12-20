from typing import Generator

import torch

from .performance_utils import Benchmark


class AttentionBenchmark(Benchmark):
    """
    benchmark for attention
    """

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def get_input_iter(self, cur_dtype) -> Generator:
        for seq_len in [1024, 2048, 3072, 4096]:
            yield from self.input_fn(cur_dtype, seq_len)


def test_perf_scaled_dot_product_attention():
    def scaled_dot_product_attention_kwargs(dtype, seq_len):
        num_heads = 8
        head_size = 128
        batch = 4

        query = torch.randn(
            (batch, num_heads, seq_len, head_size), device="cuda", dtype=dtype
        )
        key = torch.randn(
            (batch, num_heads, seq_len, head_size), device="cuda", dtype=dtype
        )
        value = torch.randn(
            (batch, num_heads, seq_len, head_size), device="cuda", dtype=dtype
        )
        yield query, key, value, None, 0.0, True

    bench = AttentionBenchmark(
        op_name="scaled_dot_product_attention",
        input_fn=scaled_dot_product_attention_kwargs,
        torch_op=torch.nn.functional.scaled_dot_product_attention,
        dtypes=[
            # torch.float32,
            torch.float16,
        ],
    )
    bench.run()
