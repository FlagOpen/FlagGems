import triton
import triton.language as tl

@triton.jit
def exp(in_ptr0, out_ptr0, N: tl.constexpr, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0_prime = offset + (loop1 * XBLOCK_SUB) + base1
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), mask=x0 < N)
        tmp2 = tl.exp(tmp0)
        tl.store(out_ptr0 + (x0), tmp2, None)
