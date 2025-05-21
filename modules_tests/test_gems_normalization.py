import numpy as np
import pytest
import torch
from torch.nn import RMSNorm

import flag_gems
from flag_gems.modules import GemsRMSNorm
from flag_gems.testing import assert_close, assert_equal

device = flag_gems.device

# TODO(flaggems): Current implementation only supports 2D input tensors and a 1D `normalized_shape` (given as int).
# Need to extend support for multi-dimensional `normalized_shape` and multi-dimensional input tensors.,
# as supported by `torch.nn.RMSNorm`.


@pytest.mark.parametrize("shape", [(4, 64), (8, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_residual", [False, True])
def test_gems_rmsnorm_matches_torch(shape, dtype, use_residual):
    np.random.seed(0)
    norm_shape = shape[-1]
    scale = 1 / (2 * norm_shape)
    np_inp = np.random.uniform(-0.1, 0.1, shape).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, norm_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=device)
    weight = torch.tensor(np_weight, dtype=dtype, device=device)
    residual = torch.randn_like(inp) * scale if use_residual else None

    reference = RMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
    reference.weight.data.copy_(weight.data)

    target = GemsRMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
    target.weight.data.copy_(weight.data)

    out_ref = reference(inp + residual) if use_residual else reference(inp)
    out_test = target(inp, residual) if use_residual else target(inp)

    assert_close(out_test, out_ref, dtype)

    if use_residual:
        torch.library.opcheck(
            torch.ops.flag_gems.fused_add_rms_norm,
            (inp, residual, target.weight.data, target.eps),
            test_utils=(
                "test_schema",
                "test_autograd_registration",
                # "test_faketensor",
                # "test_aot_dispatch_static",
                # "test_aot_dispatch_dynamic",
            ),
        )
    else:
        torch.library.opcheck(
            torch.ops.flag_gems.rms_norm,
            (inp, target.weight.data, target.eps),
            test_utils=(
                "test_schema",
                "test_autograd_registration",
                # "test_faketensor",
                # "test_aot_dispatch_static",
                # "test_aot_dispatch_dynamic",
            ),
        )
