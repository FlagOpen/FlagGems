import pytest
import torch

import flag_gems
import flag_gems.experimental

from .accuracy_utils import gems_assert_close


@pytest.mark.parametrize("shape", [(2048,), (4096,)])
@pytest.mark.parametrize("dtype", [torch.cfloat])
def test_accuracy_fft(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    y_ref = torch.fft.fft(x)
    y_res = flag_gems.experimental.fft.rad2_fft(x)
    gems_assert_close(y_ref, y_res, dtype)
