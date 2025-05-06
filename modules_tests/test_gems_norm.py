import pytest
import torch
from test_utils import assert_allclose
from torch.nn import RMSNorm

from flag_gems.modules import GemsRMSNorm


@pytest.mark.parametrize("shape", [(4, 64), (8, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("use_residual", [True, False])
def test_gems_rmsnorm_matches_torch(shape, dtype, device, use_residual):
    torch.manual_seed(42)

    x = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn_like(x) if use_residual else None

    norm_shape = shape[-1]
    reference = RMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
    target = GemsRMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)

    target.weight.data.copy_(reference.weight.data)

    out_ref = reference(x + residual) if use_residual else reference(x)
    out_test = target(x, residual)

    assert_allclose(out_ref, out_test, rtol=1e-3, atol=1e-3)
