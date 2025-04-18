import pytest
import torch

import flag_gems

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    BOOL_TYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
    unsqueeze_tensor,
    unsqueeze_tuple,
)


@pytest.mark.angle
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype",[torch.complex32]+[torch.complex64]+[torch.float32])
def test_accuracy_angle(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.angle(inp)
    with flag_gems.use_gems():
        res_out = torch.angle(inp)
    dtype_out = res_out.dtype

    gems_assert_equal(res_out, ref_out)
