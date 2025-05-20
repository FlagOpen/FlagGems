import torch
import flag_gems

def gems_assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1):
    flag_gems.testing.assert_close(
        res, ref, dtype, equal_nan=equal_nan, reduce_dim=reduce_dim
    )

def gems_assert_equal(res, ref, equal_nan=False):
    flag_gems.testing.assert_equal(res, ref, equal_nan=equal_nan)


def assert_allclose(a, b, rtol=1e-5, atol=1e-8):
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        max_diff = (a - b).abs().max().item()
        raise AssertionError(f"Tensor mismatch. Max abs diff = {max_diff}")
