import triton
import triton.language as tl

@triton.jit
def relu(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        tmp0 = tl.load(in_ptr0 + x_index, xmask)
        tmp1 = tl.load(in_ptr1 + x_index, xmask)
        tmp2 = tmp0 + libdevice.relu(tmp1)
        tl.store(out_ptr0 + x_index, tmp2, xmask)
