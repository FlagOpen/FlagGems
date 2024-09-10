import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close


@pytest.mark.parametrize("shape", [(2048,), (4096,)])
@pytest.mark.parametrize("dtype", [torch.cfloat])
def test_accuracy_fft(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cuda")
    y_ref = torch.fft.fft(x)
    with flag_gems.use_gems():
        y_res = torch.fft.fft(x)
    gems_assert_close(y_ref, y_res, dtype)
