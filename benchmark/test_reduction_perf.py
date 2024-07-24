import torch

from .performance_utils import (
    BLAS_BATCH,
    FLOAT_DTYPES,
    REDUCTION_BATCH,
    SIZES,
    Benchmark,
    unary_arg,
)


def test_perf_all():
    bench = Benchmark(
        op_name="all",
        torch_op=torch.all,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_amax():
    bench = Benchmark(
        op_name="amax",
        torch_op=torch.amax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_any():
    bench = Benchmark(
        op_name="any",
        torch_op=torch.any,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_argmax():
    bench = Benchmark(
        op_name="argmax",
        torch_op=torch.argmax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_cross_entropy_loss():
    def cross_entropy_loss_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device="cuda")
        target = torch.randint(
            0,
            size,
            [
                batch,
            ],
            device="cuda",
        )
        return inp, target

    bench = Benchmark(
        op_name="cross_entropy_loss",
        torch_op=torch.nn.CrossEntropyLoss(),
        arg_func=cross_entropy_loss_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_cumsum():
    def cumsum_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device="cuda")
        return inp, 1

    bench = Benchmark(
        op_name="cumsum",
        torch_op=torch.cumsum,
        arg_func=cumsum_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_groupnorm():
    def group_norm_args(dtype, batch, size):
        C = 16
        G = 16
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
        batch=BLAS_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_layernorm():
    def layer_norm_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device="cuda")
        weight = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device="cuda",
        )
        bias = torch.randn(
            [
                size,
            ],
            dtype=dtype,
            device="cuda",
        )
        return (
            inp,
            [
                size,
            ],
            weight,
            bias,
        )

    bench = Benchmark(
        op_name="layernorm",
        torch_op=torch.layer_norm,
        arg_func=layer_norm_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_log_softmax():
    bench = Benchmark(
        op_name="log_softmax",
        torch_op=torch.nn.functional.log_softmax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_max():
    bench = Benchmark(
        op_name="max",
        torch_op=torch.max,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_mean():
    bench = Benchmark(
        op_name="mean",
        torch_op=torch.mean,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_min():
    bench = Benchmark(
        op_name="min",
        torch_op=torch.min,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_prod():
    bench = Benchmark(
        op_name="prod",
        torch_op=torch.prod,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_softmax():
    bench = Benchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_sum():
    bench = Benchmark(
        op_name="sum",
        torch_op=torch.sum,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_var_mean():
    bench = Benchmark(
        op_name="var_mean",
        torch_op=torch.var_mean,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_vector_norm():
    bench = Benchmark(
        op_name="vector_norm",
        torch_op=torch.linalg.vector_norm,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_index_select():
    def index_select_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device="cuda")
        import random

        threshold = 0.1
        dim = random.choice([0, 1])
        index_size = inp.size(dim)
        from math import floor

        index = torch.randint(
            0, index_size, [floor(index_size * threshold)], device="cuda"
        )
        return (inp, dim, index)

    bench = Benchmark(
        op_name="index_select",
        torch_op=torch.index_select,
        arg_func=index_select_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_slice_scatter():
    def slice_scatter_args(dtype, batch, size):
        shape = [batch, size]
        inp = torch.randn(shape, dtype=dtype, device="cuda")
        import random

        dim = random.choice([0, 1])
        start = random.choice([16, 32, 64])
        end = random.choice([32, 64, 128, 256])
        step = random.choice([1, 3, 6])
        size_dim = shape[dim]
        if start is None:
            start = 0
        if end is None:
            end = size_dim
        range = end - start
        if end < start:
            range = 0
            end = start = 0
        elif (end - start) > size_dim:
            range = size_dim
            start = 0
            end = size_dim

        valid_shape = shape
        valid_shape[dim] = (range + (step - 1)) // step
        src = torch.randn(valid_shape, dtype=dtype, device="cuda")

        return (inp, src, dim, start, end, step)

    bench = Benchmark(
        op_name="slice_scatter",
        torch_op=torch.slice_scatter,
        arg_func=slice_scatter_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_select_scatter():
    def select_scatter_args(dtype, batch, size):
        shape = [batch, size]
        inp = torch.randn(shape, dtype=dtype, device="cuda")
        import random

        dim = random.choice([0, 1])
        index = random.randint(0, shape[dim])
        src_shape = shape
        del src_shape[dim]
        src = torch.randn(src_shape, dtype=dtype, device="cuda")

        return (inp, src, dim, index)

    bench = Benchmark(
        op_name="slice_scatter",
        torch_op=torch.select_scatter,
        arg_func=select_scatter_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_scatter():
    def scatter_args(dtype, batch, size):
        inp_shape = [512, 32, size // 256]
        src_shape = [128, 16, size // 256]
        inp = torch.randn(inp_shape, dtype=dtype, device="cuda")
        src = torch.randn(src_shape, dtype=dtype, device="cuda")
        import random

        dim = random.choice([0, 1, 2])
        size_dim = min(src_shape[dim], inp_shape[dim])

        index_shape = [
            random.randint(1, min(src_shape[0], inp_shape[0])),
            random.randint(1, min(src_shape[1], inp_shape[1])),
            random.randint(1, min(src_shape[2], inp_shape[2])),
        ]
        index = torch.empty(tuple(index_shape), dtype=torch.long, device="cuda")

        m, n, o = index_shape

        index_size_dim = index_shape[dim]
        # make unique indices
        for i in range(1 if dim == 0 else m):
            for j in range(1 if dim == 1 else n):
                for k in range(1 if dim == 2 else o):
                    ii = [i, j, k]
                    ii[dim] = slice(0, index.size(dim) + 1)
                    index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

        return (inp, dim, index, src)

    bench = Benchmark(
        op_name="scatter",
        torch_op=torch.scatter,
        arg_func=scatter_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_gather():
    def gather_args(dtype, batch, size):
        inp_shape = [512, 32, size // 256]
        inp = torch.randn(inp_shape, dtype=dtype, device="cuda")
        import random

        dim = random.choice([0, 1, 2])
        size_dim = inp_shape[dim]
        index_shape = [
            random.randint(1, inp_shape[0]),
            random.randint(1, inp_shape[1]),
            random.randint(1, inp_shape[2]),
        ]
        index = torch.empty(tuple(index_shape), dtype=torch.long, device="cuda")

        m, n, o = index_shape

        index_size_dim = index_shape[dim]
        # make unique indices
        for i in range(1 if dim == 0 else m):
            for j in range(1 if dim == 1 else n):
                for k in range(1 if dim == 2 else o):
                    ii = [i, j, k]
                    ii[dim] = slice(0, index.size(dim) + 1)
                    index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

        return (inp, dim, index)

    bench = Benchmark(
        op_name="gather",
        torch_op=torch.gather,
        arg_func=gather_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_index_add():
    def index_add_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device="cuda")
        import random

        dim = random.choice([0, 1])
        src_shape = list(inp.shape)
        index_max = src_shape[dim]
        index_len = index_max // 2
        index = torch.randint(0, index_max, (index_len,), device="cuda")
        src_shape[dim] = index_len
        src = torch.randn(src_shape, dtype=dtype, device="cuda")
        return (inp, dim, index, src)

    bench = Benchmark(
        op_name="index_add",
        torch_op=torch.index_add,
        arg_func=index_add_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_masked_fill():
    def masked_fill_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device="cuda")
        mask = torch.randn([batch, size], dtype=dtype, device="cuda") < 0.3
        value = 1024
        return (inp, mask, value)

    bench = Benchmark(
        op_name="masked_fill",
        torch_op=torch.masked_fill,
        arg_func=masked_fill_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.run()
