import numpy as np
import pytest
import torch
from torch.nn import RMSNorm

import flag_gems
from flag_gems.modules import GemsRMSNorm
from flag_gems.testing import assert_close

device = flag_gems.device

TEST_VLLM = True


# TODO(flaggems): Current implementation only supports 2D input tensors and a 1D `normalized_shape` (given as int).
# Need to extend support for multi-dimensional `normalized_shape` and multi-dimensional input tensors.,
# as supported by `torch.nn.RMSNorm`.
@pytest.mark.parametrize("shape", [(4, 64), (8, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gems_rmsnorm(shape, dtype):
    np.random.seed(0)
    norm_shape = shape[-1]
    np_inp = np.random.uniform(-0.1, 0.1, shape).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, norm_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=device)
    weight = torch.tensor(np_weight, dtype=dtype, device=device)

    reference = RMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
    reference.weight.data.copy_(weight.data)
    torch_ref = reference(inp)

    target = GemsRMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
    target.weight.data.copy_(weight.data)
    out_test = target(inp)
    assert_close(out_test, torch_ref, dtype, reduce_dim=norm_shape)

    if TEST_VLLM:
        from vllm.model_executor.layers.layernorm import RMSNorm as VRMSNorm

        vllm_reference = VRMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
        vllm_reference.weight.data.copy_(weight.data)
        vllm_ref = vllm_reference(inp)
        assert_close(out_test, vllm_ref, dtype, reduce_dim=norm_shape)

    torch.library.opcheck(
        torch.ops.flag_gems.rms_norm,
        (inp, target.weight.data, target.eps),
        test_utils=("test_schema", "test_autograd_registration"),
    )


@pytest.mark.parametrize("shape", [(4, 64), (8, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gems_rmsnorm_with_residual(shape, dtype):
    np.random.seed(0)
    norm_shape = shape[-1]
    scale = 1 / (2 * norm_shape)
    np_inp = np.random.uniform(-0.1, 0.1, shape).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, norm_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=device)
    weight = torch.tensor(np_weight, dtype=dtype, device=device)
    residual = torch.randn_like(inp) * scale

    reference = RMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
    reference.weight.data.copy_(weight.data)
    torch_inp = inp.clone()
    torch_residual = residual.clone()
    torch_new_residual = torch_inp + torch_residual
    torch_ref = reference(torch_new_residual)

    target = GemsRMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
    target.weight.data.copy_(weight.data)
    gems_inp = inp.clone()
    gems_residual = residual.clone()
    out_test, new_residual = target(gems_inp, gems_residual)
    assert_close(out_test, torch_ref, dtype, reduce_dim=norm_shape)
    assert_close(new_residual, torch_new_residual, dtype)

    if TEST_VLLM:
        from vllm.model_executor.layers.layernorm import RMSNorm as VRMSNorm

        vllm_reference = VRMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
        vllm_reference.weight.data.copy_(weight.data)
        vllm_inp = inp.clone()
        vllm_residual = residual.clone()
        vllm_ref, vllm_new_residual = vllm_reference(vllm_inp, vllm_residual)
        assert_close(out_test, vllm_ref, dtype, reduce_dim=norm_shape)
        assert_close(new_residual, vllm_new_residual, dtype)

    torch.library.opcheck(
        torch.ops.flag_gems.fused_add_rms_norm,
        (inp, residual, target.weight.data, target.eps),
        test_utils=("test_schema", "test_autograd_registration"),
    )
