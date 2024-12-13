import logging

import torch

from ..ops.copy import copy_


def select_scatter(inp, src, dim, index):
    logging.debug("GEMS SELECT_SCATTER")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index >= -inp.size(dim) and index < inp.size(dim), "Invalid index"
    dim = dim % inp.ndim
    index = index % inp.size(dim)

    valid_shape = list(inp.shape)
    del valid_shape[dim]
    assert (
        list(src.shape) == valid_shape
    ), "Expected src to have a size equal to the slice of self"

    out = torch.empty_strided(
        inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
    )

    out.copy_(inp)
    indices = [slice(None)] * inp.ndim
    indices[dim] = index
    copy_(src, out0=out[indices])

    return out
