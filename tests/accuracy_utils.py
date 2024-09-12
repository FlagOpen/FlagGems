import itertools

import torch

from .conftest import TO_CPU

major, minor = torch.__version__.split(".")[:2]
skip_expr = major < "2" or minor < "2"
skip_reason = "PyTorch < 2.2.0 does not support"

INT16_MIN = torch.iinfo(torch.int16).min
INT16_MAX = torch.iinfo(torch.int16).max
INT32_MIN = torch.iinfo(torch.int32).min
INT32_MAX = torch.iinfo(torch.int32).max

RESOLUTION = {
    torch.bool: 0,
    torch.int16: 0,
    torch.int32: 0,
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}

sizes_one = [1]
sizes_pow_2 = [2**d for d in range(4, 11, 2)]
sizes_noalign = [d + 17 for d in sizes_pow_2]
sizes_1d = sizes_one + sizes_pow_2 + sizes_noalign
sizes_2d_nc = [1] if TO_CPU else [1, 16, 64, 1000]
sizes_2d_nr = [1] if TO_CPU else [1, 5, 1024]

UT_SHAPES_1D = list((n,) for n in sizes_1d)
UT_SHAPES_2D = list(itertools.product(sizes_2d_nr, sizes_2d_nc))
POINTWISE_SHAPES = [(2, 19, 7)] if TO_CPU else [(1,), (1024, 1024), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29)]
SPECIAL_SHAPES = [(2, 19, 7)] if TO_CPU else [(1,), (1024, 1024), (20, 320, 15), (16, 128, 64, 1280), (16, 7, 57, 32, 29)]
DISTRIBUTION_SHAPES = [(20, 320, 15)]
REDUCTION_SHAPES = [(2, 32)] if TO_CPU else [(1, 2), (4096, 256), (200, 40999, 3)]
REDUCTION_SMALL_SHAPES = [(1, 32)] if TO_CPU else [(1, 2), (4096, 256), (200, 2560, 3)]
NONZERO_SHAPES = [(2, 32)] if TO_CPU else REDUCTION_SHAPES + [(2637,)]
CUMSUM_SHAPES = [(2, 32)] if TO_CPU else REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]
MN_SHAPES = [(1, 32)] if TO_CPU else [(1, 32), (160, 1024), (5333, 497)]
MNK_SHAPES = [(1, 1, 32)] if TO_CPU else [(1, 1, 32), (15, 160, 1024), (495, 5333, 71)]

FLIP_DIMS = [(0,), (-2,), (2,), (0, 2), (2, 1), (0, -1, 1)]
TILE_DIMS = [(0,), (2,), (2, 0), (0, 2), (2, 2), (2, 2, 2), (2, 2, 2, 2)]
REPEAT_SIZES = [(2, 3, 4, 5), (5, 0, 4)]

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
ALL_FLOAT_DTYPES = FLOAT_DTYPES + [torch.float64]
INT_DTYPES = [torch.int16, torch.int32]
ALL_INT_DTYPES = INT_DTYPES + [torch.int64]
BOOL_TYPES = [torch.bool]

SCALARS = [0.001, -0.999, 100.001, -111.999]
DIM_LIST = [1] if TO_CPU else [0, 1]
DIMS_LIST = [1] if TO_CPU else [0, 1, [0, 1], [1, 0]]

KIND_KEEPDIM_DIMS_SHAPE = [("normal", True, DIMS_LIST[0], REDUCTION_SHAPES[0])] if TO_CPU else list(zip(["normal", "allTrue"] * 2, [True, False] * 2, DIMS_LIST, REDUCTION_SHAPES + [(7, 4, 11, 1)]))
KEEPDIM_DIMS_SHAPE = [(True, DIMS_LIST[0], REDUCTION_SHAPES[0])] if TO_CPU else list(zip([True, False] * 2, DIMS_LIST, REDUCTION_SHAPES + [(7, 4, 11, 1)]))
KEEPDIM_DIMS = [(True, DIMS_LIST[0])] if TO_CPU else list(zip([True, False] * 2, DIMS_LIST))
KEEPDIM_DIM = [(True, DIM_LIST[0])] if TO_CPU else list(zip([True, False], DIM_LIST))
SMOOTH_IGNORE_SHAPE = [(0.1, 1, REDUCTION_SHAPES[0])] if TO_CPU else list(zip([0, 0.1, 1], [1, 200, -100], REDUCTION_SHAPES))
SMOOTH_SHAPE = [(0.1, REDUCTION_SHAPES[0])] if TO_CPU else list(zip([1, 0.1, 0], REDUCTION_SHAPES))
DIM_SHAPE = [(1, REDUCTION_SMALL_SHAPES[0])] if TO_CPU else list(zip([0, 1, 1], REDUCTION_SMALL_SHAPES))
THRESHOLD_SHAPE = [(0.3, REDUCTION_SHAPES[0])] if TO_CPU else list(zip([0.3, 0.5, 0.7], REDUCTION_SHAPES))
CROSS_ENTROPY_LOSS_REDUCTION = ["sum"] if TO_CPU else ["mean", "none", "sum"]


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        ref_inp = ref_inp.to(torch.float64)
    return ref_inp


def gems_assert_close(a, b, dtype, equal_nan=False, reduce_dim=1):
    if TO_CPU:
        a = a.to("cpu")
    b = b.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


def gems_assert_equal(a, b):
    if TO_CPU:
        a = a.to("cpu")
    assert torch.equal(a, b)


def unsqueeze_tuple(t, max_len):
    for _ in range(len(t), max_len):
        t = t + (1,)
    return t

def unsqueeze_tensor(inp, max_ndim):
    for _ in range(inp.ndim, max_ndim):
        inp = inp.unsqueeze(-1)
    return inp