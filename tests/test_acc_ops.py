import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close

try:
    from transformer_engine.pytorch import cpp_extensions as tex
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


GEGLU_SHAPES = [
    (4096, 1024 * 2),
    (2048, 2048 * 2),
    (1024, 4096 * 2),
    (512, 512 * 2),
    (1, 2048 * 2),
    (2048, 1 * 2),
    (512, 512, 512 * 2),
]


@pytest.mark.geglu
@pytest.mark.parametrize("shape", GEGLU_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_geglu(shape, dtype):
    if not TE_AVAILABLE:
        pytest.skip("Transformer Engine backend (cpp_extensions) not available for reference.")

    if len(shape) == 0:
        pytest.skip("GEGLU does not support 0-dim scalar tensors.")

    if shape[-1] % 2 != 0:
        shape = list(shape)
        shape[-1] += 1
        shape = tuple(shape)

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # 这里传入 quantizer=None
    ref_out = tex.geglu(input_tensor, None)

    with flag_gems.use_gems():
        res_out = flag_gems.geglu(input_tensor)

    gems_assert_close(res_out, ref_out, dtype)
