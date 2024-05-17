import triton
import triton.language as tl
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic


@pointwise_dynamic
@triton.jit
def exp_func(x):
    return tl.exp(x.to(tl.float32))


def exp(A):
    if __debug__:
        print("GEMS EXP")
    O = exp_func(A)
    return O
