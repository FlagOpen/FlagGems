import logging
import math

import torch
import triton
import triton.language as tl

from ..utils import libentry
from .topk import _get_finfo_val, _get_iinfo_val, argsort


@libentry()
@triton.jit()
def sort_kernel(
    in_ptr,
    out_ptr,
    out_index_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    in_ptr += cur_batch * N

    out_ptr += cur_batch * N
    out_index_ptr += cur_batch * N

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    if IS_FLOAT:
        mask_val = _get_finfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
    else:
        mask_val = _get_iinfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)

    if IS_FLOAT:
        in_val = tl.load(in_ptr + cols, mask=mask, other=mask_val).to(tl.float32)
    else:
        in_val = tl.load(in_ptr + cols, mask=mask, other=mask_val).to(tl.int32)

    index_val = tl.arange(0, BLOCK_SIZE).to(tl.int64)

    sorted_in_val, sorted_index_val = argsort(
        in_val, index_val, 0, descending=DESCENDING
    )

    tl.store(out_ptr + cols, sorted_in_val, mask=mask)
    tl.store(out_index_ptr + cols, sorted_index_val, mask=mask)


def sort(self, dim=-1, descending=False):
    logging.debug("GEMS SORT")
    # If dim equals to last dim, we set it to -1.
    if dim < 0:
        dim = dim + self.ndim

    sort_elem_cnt = self.shape[dim]
    batch_size = math.prod(self.shape) // sort_elem_cnt

    out = torch.empty_like(self)
    out_index = torch.empty_like(self, dtype=torch.int64)

    N = sort_elem_cnt
    BLOCK_SIZE = 512

    IS_FLOAT = self.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ]

    with torch.cuda.device(self.device):
        sort_kernel[batch_size,](
            self,
            out,
            out_index,
            N,
            BLOCK_SIZE,
            descending,
            IS_FLOAT,
        )
    return out, out_index
