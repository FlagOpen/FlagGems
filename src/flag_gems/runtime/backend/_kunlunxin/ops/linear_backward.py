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

import logging
import math

import torch

from .linear import linear
from .sum import sum_dim

logger = logging.getLogger(__name__)


def linear_backward(input, grad_output, weight, output_mask):
    logger.debug("GEMS_KUNLUNXIN LINEAR_BACKWARD")
    batch_dims = input.shape[:-1]
    # numel() // in_features divides by zero for zero-width inputs
    # (e.g. input (batch, 0), weight (out_features, 0)); derive the batch
    # size from the leading dims instead.
    batch_size = math.prod(batch_dims)
    in_features = input.shape[-1]
    out_features = weight.shape[0]
    input_flat = input.reshape(batch_size, in_features).contiguous()
    grad_output_flat = grad_output.reshape(batch_size, out_features).contiguous()

    grad_input = None
    if output_mask[0]:
        # grad_input = grad_output @ weight.  ``linear(x, W)`` is x @ W^T, so
        # feeding the transposed weight view yields exactly the GEMM above.
        # ``linear`` routes small/skinny shapes to its fused single-launch
        # kernel and the rest to the tuned ``mm``; both consume strided
        # operands, so no transpose materialisation is needed.  fp16 is left
        # native: the fused/mm kernels accumulate in fp32 and the final fp16
        # store dominates the error anyway (the fp32 upcast only added a cast
        # pass and two launches per output).
        grad_input = linear(grad_output_flat, weight.t())
        grad_input = grad_input.reshape(*batch_dims, in_features)

    grad_weight = None
    if output_mask[1]:
        # grad_weight = grad_output.T @ input.  As a ``linear`` call:
        # linear(grad_output.T, input.T) = grad_output.T @ (input.T).T.
        # The transposed views are consumed directly (no .contiguous() copy).
        grad_weight = linear(grad_output_flat.t(), input_flat.t())

    grad_bias = None
    if output_mask[2]:
        if weight.dtype in (torch.float16, torch.bfloat16):
            grad_bias = sum_dim(grad_output_flat.to(torch.float32), dim=(0,)).to(
                weight.dtype
            )
        else:
            grad_bias = sum_dim(grad_output_flat, dim=(0,))

    return grad_input, grad_weight, grad_bias
