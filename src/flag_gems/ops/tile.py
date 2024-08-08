import logging

import torch
import triton

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def tile_func(x, **kwargs):
    return x


def tile(inp: torch.Tensor, dims) -> torch.Tensor:
    logging.debug("GEMS TILE")
    inp_rank = inp.dim()
    dims_rank = len(dims)
    inp_shape = list(inp.shape)
    dims_shape = list(dims)

    if dims_rank < inp_rank:
        diff = inp_rank - dims_rank
        ones = [1 for _ in range(diff)]
        dims_shape = ones + dims_shape
    elif dims_rank > inp_rank:
        diff = dims_rank - inp_rank
        ones = [1 for _ in range(diff)]
        inp_shape = ones + inp_shape

    is_empty = False
    out_shape = []
    for i in range(len(inp_shape)):
        assert (
            dims_shape[i] >= 0
        ), "the number of repetitions per dimension out of range (expected to >= 0) but git {}".format(
            dims_shape[i]
        )
        if dims_shape[i] == 0:
            is_empty = True
        out_shape.append(inp_shape[i] * dims_shape[i])

    out = torch.empty(out_shape, device=inp.device, dtype=inp.dtype)

    inp = inp.reshape(inp_shape)
    if is_empty:
        return out
    return tile_func(inp, out0=out, in0_shape=inp_shape)
