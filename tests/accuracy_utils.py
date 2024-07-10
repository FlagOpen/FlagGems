import torch
from .conftest import TO_CPU, DEVICE
import logging

logging.debug("Using Device: " + DEVICE)

major, minor = torch.__version__.split(".")[:2]
skip_expr = major < "2" or minor < "2"
skip_reason = "PyTorch < 2.2.0 does not support"


RESOLUTION = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}

POINTWISE_SHAPES = [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)]
REDUCTION_SHAPES = [(4096, 256 * i) for i in range(1, 10, 2)]
BIG_REDUCTION_SHAPES = [(29, 32000), (27, 40960), (23, 32768, 4), (101, 16384, 11), (34, 40000, 11), (26, 910030, 11)]
MNK_SHAPES = [15, 160, 1024]

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32]

SCALARS = [0.001, -0.999, 100.001, -111.999]
DIM_LIST = [0, 1]
DIMS_LIST = [0, 1, [0, 1], [1, 0]]

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
