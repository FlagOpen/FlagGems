import torch

from .performance_utils import FLOAT_DTYPES, POINTWISE_BATCH, SIZES, Benchmark


def test_perf_embedding():
    def embedding_kwargs(dtype, batch, size):
        input = torch.randint(0, batch, (batch,), device="cuda")
        weight = torch.randn((batch + 1, size), device="cuda", dtype=dtype)
        return {"input": input, "weight": weight}

    bench = Benchmark(
        op_name="embedding",
        torch_op=torch.nn.functional.embedding,
        arg_func=None,
        dtypes=[
            torch.float32,
            torch.float16,
        ],  # Note(Zhengzekang): triton do not support bfloat16 atomic add which is used in embedding grad.
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=embedding_kwargs,
    )
    bench.run()


def test_perf_topk():
    def topk_kwargs(dtype, batch, size):
        x = torch.randn((batch, size), device="cuda", dtype=dtype)
        return {"x": x, "k": 5, "dim": -1}

    bench = Benchmark(
        op_name="topk",
        torch_op=torch.topk,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=topk_kwargs,
    )
    bench.run()


def test_perf_resolve_neg():
    def resolve_neg_arg(dtype, batch, size):
        x = torch.randn(size=(batch, size), dtype=dtype, device="cuda")
        y = x.conj()
        z = y.imag
        return (z,)

    bench = Benchmark(
        op_name="resolve_neg",
        torch_op=torch.resolve_neg,
        arg_func=resolve_neg_arg,
        dtypes=[torch.cfloat],
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_resolve_conj():
    def resolve_conj_arg(dtype, batch, size):
        x = torch.randn(size=(size, batch), dtype=dtype, device="cuda")
        return (x.conj(),)

    bench = Benchmark(
        op_name="resolve_conj",
        torch_op=torch.resolve_conj,
        arg_func=resolve_conj_arg,
        dtypes=[torch.cfloat],
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_multinomial_with_replacement():
    def multinomial_args(dtype, batch, size):
        dist = torch.rand((batch, size), dtype=dtype, device="cuda")
        n_samples = 10000
        return (dist, n_samples, True)

    bench = Benchmark(
        op_name="multinomial",
        torch_op=torch.multinomial,
        arg_func=multinomial_args,
        dtypes=(torch.float16, torch.float32),
        batch=POINTWISE_BATCH,
        sizes=(100, 1024, 2048, 4096, 8192),
    )
    bench.run()
