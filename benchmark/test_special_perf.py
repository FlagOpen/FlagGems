import torch

from .performance_utils import (
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_BATCH,
    SIZES,
    WIDE_RANGE_SIZES,
    Benchmark,
    unary_arg,
    unary_int_arg,
)


def test_perf_rand():
    def rand_kwargs(dtype, batch, size):
        return {"size": (batch, size), "dtype": dtype, "device": "cuda"}

    bench = Benchmark(
        op_name="rand",
        torch_op=torch.rand,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=rand_kwargs,
    )
    bench.run()


def test_perf_randn():
    def randn_kwargs(dtype, batch, size):
        return {"size": (batch, size), "dtype": dtype, "device": "cuda"}

    bench = Benchmark(
        op_name="randn",
        torch_op=torch.randn,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=randn_kwargs,
    )
    bench.run()


def test_perf_rand_like():
    bench = Benchmark(
        op_name="rand_like",
        torch_op=torch.rand_like,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_exponential_():
    bench = Benchmark(
        op_name="exponential_",
        torch_op=torch.Tensor.exponential_,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


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


def test_perf_unique():
    def unique_kwargs(dtype, batch, size):
        return {"sorted": True, "return_inverse": False, "return_counts": False}

    bench = Benchmark(
        op_name="unique",
        torch_op=torch.unique,
        arg_func=unary_int_arg,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=WIDE_RANGE_SIZES,
        kwargs_func=unique_kwargs,
    )
    bench.run()


def test_perf_unique_return_counts():
    def unique_kwargs(dtype, batch, size):
        return {"sorted": True, "return_inverse": False, "return_counts": True}

    bench = Benchmark(
        op_name="unique_return_counts",
        torch_op=torch.unique,
        arg_func=unary_int_arg,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=WIDE_RANGE_SIZES,
        kwargs_func=unique_kwargs,
    )
    bench.run()


def test_perf_unique_return_inverse():
    def unique_kwargs(dtype, batch, size):
        return {"sorted": True, "return_inverse": True, "return_counts": False}

    bench = Benchmark(
        op_name="unique_return_inverse",
        torch_op=torch.unique,
        arg_func=unary_int_arg,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=WIDE_RANGE_SIZES,
        kwargs_func=unique_kwargs,
    )
    bench.run()


def test_perf_unique_return_inverse_counts():
    def unique_kwargs(dtype, batch, size):
        return {"sorted": True, "return_inverse": True, "return_counts": True}

    bench = Benchmark(
        op_name="unique_return_inverse_counts",
        torch_op=torch.unique,
        arg_func=unary_int_arg,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=WIDE_RANGE_SIZES,
        kwargs_func=unique_kwargs,
    )
    bench.run()
