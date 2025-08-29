import logging

from tests.test_quant import BLOCK_SIZES
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def addcdiv_kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    lid = tl.arange(0, BLOCK_SIZE)
    offs = pid * BLOCK_SIZE + lid

    # Load data from pointers
    x = tl.load(in_ptr0 + offs)
    t1 = tl.load(in_ptr1 + offs)
    t2 = tl.load(in_ptr2 + offs)

    # Compute the operation
    result = x + in_ptr3 * (t1 / t2)

    # Store the result
    tl.store(out_ptr + offs, result)


def addcdiv(inp, tensor1, tensor2, value=1.0, out=None):
    logger.debug("GEMS ADDCDIV FORWARD")
    
    if out is None:
        out = torch.empty_like(inp)
    
    BLOCK_SIZE = 1024
    n_elements = inp.unmel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    addcdiv_kernel[grid](inp, tensor1, tensor2, value, out, BLOCK_SIZE=BLOCK_SIZE)
    return out
