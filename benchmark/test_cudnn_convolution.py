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

from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils

IS_KUNLUNXIN = flag_gems.vendor_name == "kunlunxin"


def cudnn_convolution_input_fn(shape, dtype, device):
    (
        batch,
        input_c,
        input_h,
        input_w,
        out_c,
        kernel_h,
        kernel_w,
        stride,
        padding,
        groups,
    ) = shape
    input_shape = (batch, input_c, input_h, input_w)
    weight_shape = (out_c, input_c // groups, kernel_h, kernel_w)
    inp = utils.generate_tensor_input(input_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)

    yield (
        inp,
        weight,
        [padding, padding],
        [stride, stride],
        [1, 1],
        groups,
        False,
        False,
        False,
    )


class CudnnConv2dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        shapes = [
            (32, 64, 128, 128, 32, 3, 3, 1, 2, 1),
            (32, 64, 210, 210, 16, 5, 5, 2, 1, 1),
            (16, 32, 12, 12, 24, 3, 3, 2, 1, 1),
            (16, 32, 24, 24, 24, 3, 3, 2, 2, 2),
            (16, 32, 24, 24, 24, 3, 3, 1, 2, 2),
        ]

        for shape in shapes:
            yield from self.input_fn(shape, dtype, self.device)


def _cudnn_convolution_device_baseline(
    inp, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32
):
    """SUBSTITUTED / NON-CONTRACT device baseline for ``aten::cudnn_convolution``.

    Kunlunxin-only.  ``aten::cudnn_convolution`` has no executable kernel on this
    torch_xmlir/XPU build (``Could not run 'aten::cudnn_convolution' with
    arguments from the 'CPU' backend``), so the native baseline latency cannot be
    measured with it at all and the marker would report no speedup records.
    The bias-free native device 2-D convolution computes the same quantity, runs
    on this device, and is the strongest available device reference, so it
    stands in for the missing native op.

    This is a SUBSTITUTED, NON-CONTRACT baseline: the timing reference is a
    different implementation from cuDNN.  Direction of bias -- ``F.conv2d`` is
    the vendor's own fast device convolution, i.e. the strongest available
    denominator, so the ratio below is conservative for the candidate rather
    than inflated.  It is measured at ``benchmark/base.py:754``, outside the
    ``with dispatch:`` context that wraps only the candidate call, and no
    FlagGems kernel can intercept it.
    """
    return torch.nn.functional.conv2d(
        inp, weight, None, stride, padding, dilation, groups
    )


@pytest.mark.cudnn_convolution
def test_cudnn_convolution():
    if IS_KUNLUNXIN:
        # aten::cudnn_convolution has no executable kernel on this backend, so the
        # native baseline is the equivalent bias-free device convolution, and the
        # candidate is named explicitly so both sides are measured as themselves.
        torch_op = _cudnn_convolution_device_baseline
        gems_op = flag_gems.cudnn_convolution
    else:
        torch_op = torch.ops.aten.cudnn_convolution.default
        gems_op = None
    bench = CudnnConv2dBenchmark(
        input_fn=cudnn_convolution_input_fn,
        op_name="cudnn_convolution",
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    if gems_op is not None:
        bench.set_gems(gems_op)
    bench.run()
