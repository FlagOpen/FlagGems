import logging

import triton
import triton.language as tl

from ..utils import unwrap


@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_func(input, mat1, mat2, beta=1, alpha=1):
    tmp = tl.dot(mat1, mat2)
    tmp *= alpha
    tmp1 = beta * input
    tmp += tmp1
    return tmp.to(input.type.element_ty)

def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logging.debug("GEMS ADDMM")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    mat1 = mat1.contiguous()
    mat2 = mat2.contiguous()
    return unwrap(addmm_func[(1,)](bias, mat1, mat2, beta, alpha))
