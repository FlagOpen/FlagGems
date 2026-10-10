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

import torch

logger = logging.getLogger("flag_gems.ops.row_indices_copy")


def _row_indices_copy_impl(self: torch.Tensor, *, out=None):
    """Extract ``row_indices()`` and copy it to a contiguous (or caller) tensor.

    The generic kernel decodes every destination flat offset through a
    RANK_DYN-trip stride loop and does a masked scalar gather/scatter, which on
    XPU lands on the discrete gather path (~6-7 GB/s).  The buffer is a pure
    data movement, so it is delegated to ``torch.ops.aten._copy_from`` -- the
    vendor strided-copy engine that flag_gems never overrides (~1.85 TB/s) --
    which honours the destination strides exactly like the generic kernel.
    """
    if self.layout not in (torch.sparse_csc, torch.sparse_bsc):
        raise RuntimeError(
            "row_indices expected sparse column compressed tensor layout but got "
            f"{str(self.layout).replace('torch.', '')}"
        )

    src = self.row_indices()
    if out is not None:
        if out.dtype != src.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {src.dtype}, but got {out.dtype} instead"
            )
        if out.numel() != src.numel() or out.shape != src.shape:
            out.resize_(src.shape)
        dst = out
    else:
        out_shape = tuple(src.shape)
        contig_strides = []
        stride = 1
        for dim_sz in reversed(out_shape):
            contig_strides.append(stride)
            stride *= dim_sz
        contig_strides.reverse()
        dst = torch.empty_strided(
            out_shape, contig_strides, dtype=src.dtype, device=src.device
        )

    if src.numel() > 0:
        torch.ops.aten._copy_from(src, dst, False)
    return dst


def row_indices_copy(self: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN ROW_INDICES_COPY")
    return _row_indices_copy_impl(self)


def row_indices_copy_out(self: torch.Tensor, *, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN ROW_INDICES_COPY_OUT")
    return _row_indices_copy_impl(self, out=out)
