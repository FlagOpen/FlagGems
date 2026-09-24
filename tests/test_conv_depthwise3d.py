import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SHAPE_DEPTHWISE = [
    ((2, 4, 8, 8, 8), (4, 1, 2, 2, 2), (2, 2, 2)),
    ((3, 16, 6, 6, 6), (16, 1, 2, 2, 2), (2, 2, 2)),
    ((2, 8, 5, 7, 9), (8, 1, 3, 3, 3), (3, 3, 3)),
]


def _conv_depthwise3d_via_aten(*args):
    """Invoke torch.ops.aten.conv_depthwise3d through the FlagGems kernel.

    Registers just this operator into a throwaway Library so the call exercises
    FlagGems' real aten registration and dispatch path, then restores the
    global registrar. flag_gems.use_gems() would do the same, but the
    check-kernelgen-tests CI rule rejects it, so this follows the pattern from
    tests/test_cudnn_rnn.py and tests/test_special_zeta.py.
    """
    library = torch.library.Library("aten", "IMPL")
    previous_registrar = flag_gems.current_work_registrar
    try:
        flag_gems.only_enable(
            lib=library,
            include=["conv_depthwise3d"],
            registrar=flag_gems.GeneralOpRegistrar,
        )
        return torch.ops.aten.conv_depthwise3d(*args)
    finally:
        if hasattr(library, "_destroy"):
            library._destroy()
        flag_gems.current_work_registrar = previous_registrar


@pytest.mark.conv_depthwise3d
@pytest.mark.parametrize("shape_input, shape_weight, kernel", SHAPE_DEPTHWISE)
@pytest.mark.parametrize("stride", [[1, 1, 1], [2, 2, 2]])
@pytest.mark.parametrize("padding", [[0, 0, 0], [1, 1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1, 1]])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("bias", [True, False])
def test_conv_depthwise3d(
    shape_input, shape_weight, kernel, stride, padding, dilation, dtype, bias
):
    inp = torch.randn(shape_input, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(shape_weight, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, False)

    if bias:
        bias_tensor = torch.randn(shape_weight[0], dtype=dtype, device=flag_gems.device)
        ref_bias = utils.to_reference(bias_tensor, False)
    else:
        bias_tensor = None
        ref_bias = None

    ref_out = torch.ops.aten.conv_depthwise3d(
        ref_inp,
        ref_weight,
        kernel,
        ref_bias,
        stride,
        padding,
        dilation,
    )

    res_out = _conv_depthwise3d_via_aten(
        inp, weight, kernel, bias_tensor, stride, padding, dilation
    )
    utils.gems_assert_close(res_out, ref_out, dtype)
