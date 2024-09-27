import itertools

import pytest
import torch

from .attri_util import DEFAULT_NON_BLAS_BENCH_SHAPES, BenchLevel
from .conftest import Config
from .performance_utils import (
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_BATCH,
    Benchmark,
    binary_args,
    ternary_args,
    unary_arg,
)

POINTWISE_SHAPES = DEFAULT_NON_BLAS_BENCH_SHAPES[:]
if Config.bench_level == BenchLevel.COMPREHENSIVE:
    MORE_SHAPES = [(320, 15), (128, 64, 60)]
    MORE_BATCHS = [4, 20, 32]
    combinations = [
        (batch, *shape) for batch, shape in itertools.product(MORE_BATCHS, MORE_SHAPES)
    ]
    POINTWISE_SHAPES.extend(combinations)


@pytest.mark.abs
def test_perf_abs():
    bench = Benchmark(
        op_name="abs",
        torch_op=torch.abs,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.add
def test_perf_add():
    bench = Benchmark(
        op_name="add",
        torch_op=torch.add,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.bitwise_and
def test_perf_bitwiseand():
    bench = Benchmark(
        op_name="bitwiseand_int",
        torch_op=torch.bitwise_and,
        arg_func=binary_args,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.bitwise_not
def test_perf_bitwisenot():
    bench = Benchmark(
        op_name="bitwisenot_int",
        torch_op=torch.bitwise_not,
        arg_func=unary_arg,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.bitwise_or
def test_perf_bitwiseor():
    bench = Benchmark(
        op_name="bitwiseor_int",
        torch_op=torch.bitwise_or,
        arg_func=binary_args,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.clamp
def test_perf_clamp():
    bench = Benchmark(
        op_name="clamp",
        torch_op=torch.clamp,
        arg_func=ternary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.cos
def test_perf_cos():
    bench = Benchmark(
        op_name="cos",
        torch_op=torch.cos,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.div
def test_perf_div():
    bench = Benchmark(
        op_name="div",
        torch_op=torch.div,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.floor_divide
def test_perf_floordiv_int():
    bench = Benchmark(
        op_name="floor_divide",
        torch_op=torch.floor_divide,
        arg_func=binary_args,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.remainder
def test_perf_remainder():
    bench = Benchmark(
        op_name="remainder",
        torch_op=torch.remainder,
        arg_func=binary_args,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


# TODO: enable @pytest.mark.native_dropout or not
@pytest.mark.dropout
def test_perf_dropout():
    bench = Benchmark(
        op_name="dropout",
        torch_op=torch.nn.Dropout(p=0.5),
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.eq
def test_perf_eq():
    bench = Benchmark(
        op_name="eq",
        torch_op=torch.eq,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.maximum
def test_perf_maximum():
    bench = Benchmark(
        op_name="maximum",
        torch_op=torch.maximum,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.minimum
def test_perf_minimum():
    bench = Benchmark(
        op_name="minimum",
        torch_op=torch.minimum,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.exp
def test_perf_exp():
    bench = Benchmark(
        op_name="exp",
        torch_op=torch.exp,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.ge
def test_perf_ge():
    bench = Benchmark(
        op_name="ge",
        torch_op=torch.ge,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.gelu
def test_perf_gelu():
    bench = Benchmark(
        op_name="gelu",
        torch_op=torch.nn.functional.gelu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.gelu_backward
def test_perf_gelu_backward():
    bench = Benchmark(
        op_name="gelu",
        torch_op=torch.nn.functional.gelu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
        is_backward=True,
    )
    bench.run()


@pytest.mark.gt
def test_perf_gt():
    bench = Benchmark(
        op_name="gt",
        torch_op=torch.gt,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.isinf
def test_perf_isinf():
    bench = Benchmark(
        op_name="isinf",
        torch_op=torch.isinf,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.isnan
def test_perf_isnan():
    bench = Benchmark(
        op_name="isnan",
        torch_op=torch.isnan,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.le
def test_perf_le():
    bench = Benchmark(
        op_name="le",
        torch_op=torch.le,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.lt
def test_perf_lt():
    bench = Benchmark(
        op_name="lt",
        torch_op=torch.lt,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.mul
def test_perf_mul():
    bench = Benchmark(
        op_name="mul",
        torch_op=torch.mul,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.ne
def test_perf_ne():
    bench = Benchmark(
        op_name="ne",
        torch_op=torch.ne,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.neg
def test_perf_neg():
    bench = Benchmark(
        op_name="neg",
        torch_op=torch.neg,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.pow
def test_perf_pow():
    bench = Benchmark(
        op_name="pow",
        torch_op=torch.pow,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.reciprocal
def test_perf_reciprocal():
    bench = Benchmark(
        op_name="reciprocal",
        torch_op=torch.reciprocal,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.relu
def test_perf_relu():
    bench = Benchmark(
        op_name="relu",
        torch_op=torch.nn.functional.relu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.rsqrt
def test_perf_rsqrt():
    bench = Benchmark(
        op_name="rsqrt",
        torch_op=torch.rsqrt,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.sigmoid
def test_perf_sigmoid():
    bench = Benchmark(
        op_name="sigmoid",
        torch_op=torch.sigmoid,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.silu
def test_perf_silu():
    bench = Benchmark(
        op_name="silu",
        torch_op=torch.nn.functional.silu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.sin
def test_perf_sin():
    bench = Benchmark(
        op_name="sin",
        torch_op=torch.sin,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.sub
def test_perf_sub():
    bench = Benchmark(
        op_name="sub",
        torch_op=torch.sub,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.tanh
def test_perf_tanh():
    bench = Benchmark(
        op_name="tanh",
        torch_op=torch.tanh,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.triu
def test_perf_triu():
    bench = Benchmark(
        op_name="triu",
        torch_op=torch.triu,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.where
def test_perf_where():
    def where_args(dtype, batch, shape):
        inp1 = torch.randn(shape, dtype=dtype, device="cuda")
        inp2 = torch.randn(shape, dtype=dtype, device="cuda")
        condition = inp1 > 0
        return condition, inp1, inp2

    bench = Benchmark(
        op_name="where",
        torch_op=torch.where,
        arg_func=where_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.isclose
def test_perf_isclose():
    bench = Benchmark(
        op_name="isclose",
        torch_op=torch.isclose,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES + INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.allclose
def test_perf_allclose():
    bench = Benchmark(
        op_name="allclose",
        torch_op=torch.allclose,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES + INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.erf
def test_perf_erf():
    bench = Benchmark(
        op_name="erf",
        torch_op=torch.erf,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.isfinite
def test_perf_isfinite():
    bench = Benchmark(
        op_name="isfinite",
        torch_op=torch.isfinite,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES + INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.flip
def test_perf_flip():
    def flip_kwargs(dtype, batch, shape):
        return {"dims": [0, 1]}

    bench = Benchmark(
        op_name="flip",
        torch_op=torch.flip,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES + INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
        kwargs_func=flip_kwargs,
    )
    bench.run()


@pytest.mark.masked_fill
def test_masked_fill():
    def masked_fill_args(dtype, batch, shape):
        inp = torch.randn(shape, dtype=dtype, device="cuda")
        mask = torch.randn(shape, dtype=dtype, device="cuda") < 0.3
        value = 1024
        return (inp, mask, value)

    bench = Benchmark(
        op_name="masked_fill",
        torch_op=torch.masked_fill,
        arg_func=masked_fill_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()


@pytest.mark.tile
def test_perf_tile():
    def tile_kwargs(dtype, batch, shape):
        return {"dims": [2, 4]}

    bench = Benchmark(
        op_name="tile",
        torch_op=torch.tile,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
        kwargs_func=tile_kwargs,
    )
    bench.run()


REPEAT_SHAPES = DEFAULT_NON_BLAS_BENCH_SHAPES[:]


@pytest.mark.repeat
def test_perf_repeat():
    def repeat_arg(dtype, batch, shape):
        inp1 = torch.randn(shape, dtype=dtype, device="cuda")
        inp2 = [2, 4]
        print("inp1", shape)
        return inp1, inp2

    bench = Benchmark(
        op_name="repeat",
        torch_op=torch.Tensor.repeat,
        arg_func=repeat_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=POINTWISE_SHAPES,
    )
    bench.run()
