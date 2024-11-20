import flag_gems 
import torch
import logging
logging.basicConfig(level="DEBUG")

TO_CPU = None
def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if upcast:
        ref_inp = ref_inp.to(torch.float64)
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    return ref_inp

RESOLUTION = {
    torch.bool: 0,
    torch.int16: 0,
    torch.int32: 0,
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}
QUICK_MODE = False
POINTWISE_SHAPES = (
    [(2, 19, 7)]
    if QUICK_MODE
    else [(1,), (1024, 1024), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29)]
)
TILE_DIMS = [ (2,), (2, 0), (0, 2), (2, 2), (2, 2, 2), (2, 2, 2, 2)]
FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
def assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1):
    assert res.dtype == dtype
    ref = ref.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(res, ref, atol=atol, rtol=rtol, equal_nan=equal_nan)

def test_accuracy_tile(shape, dims, dtype):
    inp = torch.randn(shape, dtype=dtype, device='cuda')
    ref_inp = to_reference(inp)
    ref_out = torch.tile(ref_inp, dims)
    with flag_gems.use_gems():
        res_out = torch.tile(inp, dims)

    assert_close(res_out, ref_out, dtype)


for dim in TILE_DIMS:
    for shape in POINTWISE_SHAPES:
        for type in FLOAT_DTYPES:
            print(shape, dim, type)
            test_accuracy_tile(shape, dim, type)
