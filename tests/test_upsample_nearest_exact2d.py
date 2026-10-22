import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.upsample_nearest_exact2d
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("factor", [2, 3])
def test_upsample_nearest_exact2d(shape, dtype, factor):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    out_size = [shape[2] * factor, shape[3] * factor]
    ref_out = torch.ops.aten._upsample_nearest_exact2d(ref_x, out_size, None, None)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._upsample_nearest_exact2d(x, out_size, None, None)
    utils.gems_assert_close(res_out, ref_out, dtype)
