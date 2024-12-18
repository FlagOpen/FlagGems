import logging

import torch
import triton

from ..ops.copy import copy
from ..utils.shape_utils import MemOverlap, has_internal_overlapping


def slice_scatter(inp, src, dim=0, start=None, end=None, step=1):
    logging.debug("GEMS SLICE_SCATTER")
    assert src.device == inp.device, "inp and src reside on different devices."
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert step > 0, "slice step must be positive"
    dim = dim % inp.ndim

    start = start or 0
    end = end or inp.size(dim)
    if end < 0:
        end = end % inp.size(dim)

    valid_shape = list(inp.shape)
    valid_shape[dim] = triton.cdiv(end - start, step)
    assert (
        list(src.shape) == valid_shape
    ), "Expected src to have a size equal to the slice of self"

    if has_internal_overlapping(inp) == MemOverlap.Yes:
        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
    else:
        out = torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )

    ndim = inp.ndim
    copy(inp, out0=out)

    indices = [slice(None)] * ndim
    indices[dim] = slice(start, end, step)
    out_ = out[indices]
    copy(src, out0=out_)

    return out
