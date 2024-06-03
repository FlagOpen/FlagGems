import torch
import pytest
import flag_gems
from .accuracy_utils import *


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rsub(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="musa")
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = torch.rsub(ref_inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.rsub(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pow_scalar_tensor(scalar, shape, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="musa")
    ref_inp2 = to_reference(inp2, False)

    ref_out = torch.pow(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.pow(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)

