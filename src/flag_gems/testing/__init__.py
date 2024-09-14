import torch

RESOLUTION = {
    torch.bool: 0,
    torch.int16: 0,
    torch.int32: 0,
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}


def assert_close(a, b, dtype, equal_nan=False, reduce_dim=1):
    b = b.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


def assert_equal(a, b):
    assert torch.equal(a, b)
