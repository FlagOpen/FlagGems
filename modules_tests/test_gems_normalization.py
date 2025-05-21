import pytest
import torch
from  flag_gems.testing import assert_close, assert_equal
from torch.nn import RMSNorm
import numpy as np


from flag_gems.modules import GemsRMSNorm


@pytest.mark.parametrize("shape", [(4, 64), (8, 128)]) # (4, 6, 64)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("use_residual", [False]) # True, 
def test_gems_rmsnorm_matches_torch(shape, dtype, device, use_residual):

    # np.random.seed(0)
    # norm_shape = shape[-1]
    # np_inp = np.random.uniform(-0.1, 0.1, shape).astype(np.float32)
    # np_weight = np.random.uniform(-0.1, 0.1, norm_shape).astype(np.float32)

    # inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device, requires_grad=True)
    # weight = torch.tensor(
    #     np_weight, dtype=dtype, device=flag_gems.device, requires_grad=True
    # )

    torch.manual_seed(0)
    norm_shape = shape[-1]
    scale = 1 / (2 * norm_shape)

    x = torch.randn(shape, device=device, dtype=dtype)
    x *= scale
    residual = torch.randn_like(x) * scale if use_residual else None

    reference = RMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
    reference.weight.data.normal_(mean=1.0, std=0.1)

    target = GemsRMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
    target.weight.data.copy_(reference.weight.data)

    out_ref = reference(x + residual) if use_residual else reference(x)
    out_test = target(x, residual) if use_residual else target(x)

    assert_close(out_test, out_ref, dtype)

    #assert_allclose(out_ref, out_test, rtol=1e-3, atol=1e-3)

    if use_residual:
        torch.library.opcheck(
            torch.ops.flag_gems.fused_add_rms_norm,
            (x, residual, target.weight.data, target.eps),
            test_utils=(
            "test_schema",
            #"test_autograd_registration",
            #"test_faketensor",
            # "test_aot_dispatch_static",
            # "test_aot_dispatch_dynamic",
            ))
    else:
        torch.library.opcheck(
            torch.ops.flag_gems.rms_norm,
            (x, target.weight.data, target.eps),
            test_utils=(
            "test_schema",
            # "test_autograd_registration",
            # "test_faketensor",
            # "test_aot_dispatch_static",
            # "test_aot_dispatch_dynamic",
            ))
