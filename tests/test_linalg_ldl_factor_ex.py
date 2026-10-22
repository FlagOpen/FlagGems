import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.fixture(autouse=True)
def _candidate_aliases(request, monkeypatch):
    """Route the public operator override through this suite's native entrypoint."""
    registry = getattr(request.config, "_override_registry", None)
    overridden = set()
    if registry is None:
        return overridden
    for op_name, symbol in (
        ("linalg_ldl_factor_ex", "ldl_factor_ex"),
    ):
        candidate = registry.get_override(op_name)
        if candidate is not None:
            monkeypatch.setattr(flag_gems, symbol, candidate)
            overridden.add(op_name)
    return overridden

# Worktree test uses square matrices for LDL factorization.
LDLD_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32)]
# PyTorch linalg_ldl_factor_ex only supports float32 on CUDA.
LDLD_DTYPES = [torch.float32]


@pytest.mark.linalg_ldl_factor_ex
@pytest.mark.parametrize("shape", LDLD_SHAPES)
@pytest.mark.parametrize("dtype", LDLD_DTYPES)
def test_linalg_ldl_factor_ex(shape, dtype):
    # LDL factorization requires symmetric positive definite matrices
    # Create a random matrix and make it symmetric positive definite
    n = shape[-1]
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Make it symmetric: A = A @ A^T + small diagonal for numerical stability
    A = A @ A.mT + torch.eye(n, dtype=dtype, device=flag_gems.device) * 0.1

    ref_A = utils.to_reference(A, True)

    ref_out = torch.linalg.ldl_factor_ex(ref_A)
    with flag_gems.use_gems():
        res_out = torch.linalg.ldl_factor_ex(A)

    # Compare LD matrices
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    # Compare pivots
    utils.gems_assert_equal(res_out[1], ref_out[1])
    # Compare info (should be 0 for successful factorization)
    utils.gems_assert_equal(res_out[2], ref_out[2])
