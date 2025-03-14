import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit
def mv_func(inp, vec, n_elements: tl.constexpr):
    reshaped_vec = tl.reshape(vec, (n_elements, 1))
    out = tl.dot(inp, reshaped_vec, out_dtype=inp.type.element_ty)
    return out


def mv(inp, vec):
    logging.debug("GEMS MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"

    n_elements = vec.shape[0]
    return unwrap(mv_func[(1,)](inp, vec, n_elements)).reshape(-1)
