import triton
import triton.language as tl

@triton.jit
def neg(in_ptr0, out_ptr0, N: tl.constexpr, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index, mask=x_index < N)
        tmp1 = -tmp0
        tl.store(out_ptr0 + x_index, tmp1, mask=x_index < N)
