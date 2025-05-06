import torch


def assert_allclose(a, b, rtol=1e-5, atol=1e-8):
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        max_diff = (a - b).abs().max().item()
        raise AssertionError(f"Tensor mismatch. Max abs diff = {max_diff}")
