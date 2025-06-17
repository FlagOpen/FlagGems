import pytest
import torch
import flag_gems
import itertools
from .accuracy_utils import (
    FLOAT_DTYPES,
    SCALARS,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
M_VALUES = [1, 33, 64, 222]
TOP_KS = [2, 6]
K_VALUES = [128, 511, 1024]
MOE_SHAPES = list(itertools.product(M_VALUES, TOP_KS, K_VALUES))

@pytest.mark.parametrize("shape", MOE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_moe_sum(shape, dtype):
    m, topk, k = shape
    inp1 = torch.randn((m, topk, k), dtype=dtype, device=flag_gems.device)
    res_out = torch.empty((m, k), dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_out = torch.sum(ref_inp1, dim=1)
    with flag_gems.use_gems():
        flag_gems.moe_sum(inp1, res_out)
    gems_assert_close(res_out, ref_out, dtype)