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

from . import accuracy_utils as utils


@pytest.mark.upsample_trilinear3d_backward
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "scale", [(2, 2, 2), (1.5, 2.1, 3.7), (0.5, 0.5, 0.5), (0.3, 1.3, 0.7)]
)
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES_3D)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_trilinear3d_backward(dtype, shape, scale, align_corners):
    torch.manual_seed(0)
    input_size = list(shape)
    output_size = [max(1, int(shape[i + 2] * scale[i])) for i in range(3)]

    grad_output = torch.randn(
        [shape[0], shape[1], *output_size], dtype=dtype, device=flag_gems.device
    )
    ref_grad = utils.to_reference(grad_output, True).to(torch.float32)

    ref_out = torch.ops.aten.upsample_trilinear3d_backward.default(
        ref_grad, output_size, input_size, align_corners, None, None, None
    ).to(dtype)
    res_out = flag_gems.upsample_trilinear3d_backward(
        grad_output, output_size, input_size, align_corners, None, None, None
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
