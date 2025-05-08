import triton
import triton.language as tl

@triton.jit
def silu(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block, mask=idx_block < N)
    ret = x * (1/(1+libdevice.exp(-x)))
    tl.store(out_ptr0 + idx_block, ret, mask=idx_block < N)
