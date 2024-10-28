import itertools

import torch

import flag_gems

from .conftest import QUICK_MODE, TO_CPU


def SkipTorchVersion(skip_pattern):
    cmp = skip_pattern[0]
    assert cmp in ("=", "<", ">")
    try:
        M, N = skip_pattern[1:].split(".")
        M, N = int(M), int(N)
    except Exception:
        raise "Cannot parse version number."
    major, minor = torch.__version__.split(".")[:2]
    major, minor = int(major), int(minor)

    if cmp == "=":
        return major == M and minor == N
    elif cmp == "<":
        return (major, minor) < (M, N)
    else:
        return (major, minor) > (M, N)


INT16_MIN = torch.iinfo(torch.int16).min
INT16_MAX = torch.iinfo(torch.int16).max
INT32_MIN = torch.iinfo(torch.int32).min
INT32_MAX = torch.iinfo(torch.int32).max

sizes_one = [1]
sizes_pow_2 = [2**d for d in range(4, 11, 2)]
sizes_noalign = [d + 17 for d in sizes_pow_2]
sizes_1d = sizes_one + sizes_pow_2 + sizes_noalign
sizes_2d_nc = [1] if QUICK_MODE else [1, 16, 64, 1000]
sizes_2d_nr = [1] if QUICK_MODE else [1, 5, 1024]

UT_SHAPES_1D = list((n,) for n in sizes_1d)
UT_SHAPES_2D = list(itertools.product(sizes_2d_nr, sizes_2d_nc))
POINTWISE_SHAPES = (
    [(2, 19, 7)]
    if QUICK_MODE
    else [(), (1,), (1024, 1024), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29)]
)
SPECIAL_SHAPES = (
    [(2, 19, 7)]
    if QUICK_MODE
    else [(1,), (1024, 1024), (20, 320, 15), (16, 128, 64, 1280), (16, 7, 57, 32, 29)]
)
DISTRIBUTION_SHAPES = [(20, 320, 15)]
REDUCTION_SHAPES = [(2, 32)] if QUICK_MODE else [(1, 2), (4096, 256), (200, 40999, 3)]
REDUCTION_SMALL_SHAPES = (
    [(1, 32)] if QUICK_MODE else [(1, 2), (4096, 256), (200, 2560, 3)]
)
STACK_SHAPES = [
    [(16,), (16,)],
    [(16, 256), (16, 256)],
    [(20, 320, 15), (20, 320, 15), (20, 320, 15)],
]


UPSAMPLE_SHAPES = [
    (32, 16, 128, 128),
    (15, 37, 256, 256),
    (3, 5, 127, 127),
    (128, 192, 42, 51),
    (3, 7, 1023, 1025),
]


FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
ALL_FLOAT_DTYPES = FLOAT_DTYPES + [torch.float64]
INT_DTYPES = [torch.int16, torch.int32]
ALL_INT_DTYPES = INT_DTYPES + [torch.int64]
BOOL_TYPES = [torch.bool]

SCALARS = [0.001, -0.999, 100.001, -111.999]
STACK_DIM_LIST = [-2, -1, 0, 1]


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if upcast:
        ref_inp = ref_inp.to(torch.float64)
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    return ref_inp


def to_cpu(res, ref):
    if TO_CPU:
        res = res.to("cpu")
        assert ref.device == torch.device("cpu")
    return res


def gems_assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1):
    res = to_cpu(res, ref)
    flag_gems.testing.assert_close(
        res, ref, dtype, equal_nan=equal_nan, reduce_dim=reduce_dim
    )


def gems_assert_equal(res, ref, equal_nan=False):
    res = to_cpu(res, ref)
    flag_gems.testing.assert_equal(res, ref, equal_nan=equal_nan)


def unsqueeze_tuple(t, max_len):
    for _ in range(len(t), max_len):
        t = t + (1,)
    return t


def unsqueeze_tensor(inp, max_ndim):
    for _ in range(inp.ndim, max_ndim):
        inp = inp.unsqueeze(-1)
    return inp
