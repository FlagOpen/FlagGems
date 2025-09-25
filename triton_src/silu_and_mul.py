"""Triton kernel for fused SiLU activation followed by element-wise multiplication."""

import triton
import triton.language as tl


@triton.jit
def silu_and_mul_kernel(X, Y, OUT, n, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n

    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)

    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)

    silu = x_fp32 / (1 + tl.exp(-x_fp32))
    out = (silu * y_fp32).to(OUT.dtype.element_ty)

    tl.store(OUT + offsets, out, mask=mask)
