import triton
import triton.language as tl

@triton.jit
def bitwise_or(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index, mask=x_index < N)
        tmp1 = tl.load(in_ptr1 + x_index, mask=x_index < N)
        tmp2 = tmp0 | tmp1
        tl.store(out_ptr0 + x_index, tmp2, mask=x_index < N)
