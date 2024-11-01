import torch

from .performance_utils import Benchmark


def test_perf_scaled_dot_product_attention():
    def scaled_dot_product_attention_kwargs(dtype, batch, seq_len):
        num_heads = 8
        head_size = 128

        query = torch.randn(
            (batch, num_heads, seq_len, head_size), device="cuda", dtype=dtype
        )
        key = torch.randn(
            (batch, num_heads, seq_len, head_size), device="cuda", dtype=dtype
        )
        value = torch.randn(
            (batch, num_heads, seq_len, head_size), device="cuda", dtype=dtype
        )
        return {"query": query, "key": key, "value": value, "is_causal": True}

    seq_len = [1024, 2048, 3072, 4096]
    bench = Benchmark(
        op_name="scaled_dot_product_attention",
        torch_op=torch.nn.functional.scaled_dot_product_attention,
        arg_func=None,
        dtypes=[
            # torch.float32,
            torch.float16,
        ],
        batch=4,
        sizes=seq_len,
        kwargs_func=scaled_dot_product_attention_kwargs,
    )
    bench.run()
