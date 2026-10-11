# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close
from .conftest import QUICK_MODE

IS_KUNLUNXIN = flag_gems.vendor_name == "kunlunxin"


def _cudnn_convolution_device_ref(input, weight, padding, stride, dilation, groups):
    """Device golden reference for ``aten::cudnn_convolution`` (kunlunxin only).

    ``aten::cudnn_convolution`` has no executable kernel on this torch_xmlir/XPU
    build: every call raises ``NotImplementedError: Could not run
    'aten::cudnn_convolution' with arguments from the 'CPU' backend``, so the
    upstream reference call can never produce a golden output.  Rebuild the
    reference from the equivalent *device* op
    ``torch.nn.functional.conv{1,2,3}d``, which is the plain aten convolution
    running on the same device.  The mapping is 1:1: ``cudnn_convolution`` takes
    no bias, its ``padding``/``stride``/``dilation`` are per spatial dimension,
    and ``groups`` is passed through unchanged.  No FlagGems code is involved, so
    the candidate is still compared against an independent implementation.
    """
    conv = {
        1: torch.nn.functional.conv1d,
        2: torch.nn.functional.conv2d,
        3: torch.nn.functional.conv3d,
    }[input.ndim - 2]
    return conv(
        input, weight, None, list(stride), list(padding), list(dilation), groups
    )


if QUICK_MODE:
    SHAPE_CUDNN_CONV2D = [
        ((1, 2, 5, 5), (1, 2, 3, 3), 1),
    ]
    FLOAT_DTYPES = [torch.float32]

else:
    SHAPE_CUDNN_CONV2D = [
        ((1, 2, 5, 5), (1, 2, 3, 3), 1),
        ((2, 3, 9, 9), (1, 3, 3, 3), 1),
        # ((32, 8, 8, 8), (32, 8, 2, 2), 1),  # commented out to reduce CI timeout
    ]
    FLOAT_DTYPES = [torch.float16, torch.float32]


@pytest.mark.cudnn_convolution
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CUDNN_CONV2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dilation", [1, 2])
def test_cudnn_convolution_2d(
    shape, kernel, stride, padding, groups, dtype, dilation, monkeypatch
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)

    if IS_KUNLUNXIN:
        ref_out = _cudnn_convolution_device_ref(
            inp,
            weight,
            [padding, padding],
            [stride, stride],
            [dilation, dilation],
            groups,
        )
    else:
        ref_out = torch.cudnn_convolution(
            inp,
            weight,
            padding=[padding, padding],
            stride=[stride, stride],
            dilation=[dilation, dilation],
            groups=groups,
            benchmark=False,
            deterministic=False,
            allow_tf32=False,
        )

    res_out = flag_gems.cudnn_convolution(
        inp,
        weight,
        padding=[padding, padding],
        stride=[stride, stride],
        dilation=[dilation, dilation],
        groups=groups,
        benchmark=False,
        deterministic=False,
        allow_tf32=False,
    )

    gems_assert_close(res_out.cpu(), ref_out.cpu(), dtype)


if QUICK_MODE:
    SHAPE_CUDNN_CONV1D = [
        ((32, 2, 4), (17, 2, 2)),
    ]
else:
    SHAPE_CUDNN_CONV1D = [
        ((32, 2, 4), (17, 2, 2)),
        ((32, 15, 6), (17, 15, 2)),
        # ((64, 64, 64), (128, 64, 7)),  # commented out to reduce CI timeout - large
    ]


@pytest.mark.cudnn_convolution
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.parametrize("shape, kernel", SHAPE_CUDNN_CONV1D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cudnn_convolution_1d(shape, kernel, stride, padding, dtype, monkeypatch):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)

    if IS_KUNLUNXIN:
        ref_out = _cudnn_convolution_device_ref(
            inp,
            weight,
            [padding],
            [stride],
            [1],
            1,
        )
    else:
        ref_out = torch.cudnn_convolution(
            inp,
            weight,
            padding=[padding],
            stride=[stride],
            dilation=[1],
            groups=1,
            benchmark=False,
            deterministic=False,
            allow_tf32=False,
        )

    res_out = flag_gems.cudnn_convolution(
        inp,
        weight,
        padding=[padding],
        stride=[stride],
        dilation=[1],
        groups=1,
        benchmark=False,
        deterministic=False,
        allow_tf32=False,
    )

    gems_assert_close(res_out.cpu(), ref_out.cpu(), dtype)


if QUICK_MODE:
    SHAPE_CUDNN_CONV3D = [
        ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), 1),
    ]
else:
    SHAPE_CUDNN_CONV3D = [
        ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), 1),
        # ((2, 3, 9, 9, 9), (1, 3, 3, 3, 3), 1),  # commented out to reduce CI timeout
    ]


@pytest.mark.cudnn_convolution
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CUDNN_CONV3D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dilation", [1, 2])
def test_cudnn_convolution_3d(
    shape, kernel, stride, padding, groups, dtype, dilation, monkeypatch
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)

    if IS_KUNLUNXIN:
        ref_out = _cudnn_convolution_device_ref(
            inp,
            weight,
            [padding, padding, padding],
            [stride, stride, stride],
            [dilation, dilation, dilation],
            groups,
        )
    else:
        ref_out = torch.cudnn_convolution(
            inp,
            weight,
            padding=[padding, padding, padding],
            stride=[stride, stride, stride],
            dilation=[dilation, dilation, dilation],
            groups=groups,
            benchmark=False,
            deterministic=False,
            allow_tf32=False,
        )

    res_out = flag_gems.cudnn_convolution(
        inp,
        weight,
        padding=[padding, padding, padding],
        stride=[stride, stride, stride],
        dilation=[dilation, dilation, dilation],
        groups=groups,
        benchmark=False,
        deterministic=False,
        allow_tf32=False,
    )

    gems_assert_close(res_out.cpu(), ref_out.cpu(), dtype)
