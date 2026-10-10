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
from typing import List, Tuple, Union

import torch

logger = logging.getLogger(__name__)


def _reshape_input(t: torch.Tensor) -> torch.Tensor:
    # Zero or one dimensional tensors are reshaped into a (numel, 1) column.
    if t.ndim <= 1:
        return t.reshape(t.numel(), 1)
    return t


def column_stack_out(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN COLUMN_STACK_OUT")

    if len(tensors) == 0:
        raise RuntimeError("column_stack expected a non-empty TensorList")

    reshaped = [_reshape_input(t) for t in tensors]

    inp0_shape = list(reshaped[0].shape)
    dim = 1

    for tensor_num, tensor in enumerate(reshaped[1:]):
        if tensor.ndim != reshaped[0].ndim:
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got "
                f"{reshaped[0].ndim} and {tensor.ndim}"
            )
        inp_shape = list(tensor.shape)
        for i in range(len(inp_shape)):
            if i != dim and inp_shape[i] != inp0_shape[i]:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {inp0_shape[i]} but got size {inp_shape[i]} "
                    f"for tensor number {tensor_num + 1} in the list."
                )

    # Type promotion: find the common dtype for all tensors.
    dtype = reshaped[0].dtype
    for t in reshaped[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    reshaped = [t.to(dtype) if t.dtype != dtype else t for t in reshaped]

    out_shape = list(inp0_shape)
    out_shape[dim] = sum(int(t.shape[dim]) for t in reshaped)

    out = out.view(out_shape)

    # column_stack always concatenates along dim=1 (inner dim). Each source is
    # contiguous and maps to a strided column slab of `out`. Use the native ATen
    # strided-copy engine (`_copy_from` is the device DMA primitive, never
    # overridden by gems) exactly like the inner-dim branch of cat_out; a
    # hand-written Triton gather-store kernel (the generic path) measures
    # ~38.7ms per tensor here because every store is a non-affine scatter.
    start = 0
    for tensor in reshaped:
        w = int(tensor.shape[dim])
        if tensor.numel() > 0:
            torch.ops.aten._copy_from(tensor.contiguous(), out.narrow(dim, start, w), False)
        start += w

    return out
