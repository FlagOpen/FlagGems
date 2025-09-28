import torch
import triton
import triton.language as tl

@triton.jit
def atan_kernel(
    x_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # atan(-x) = -atan(x)
    sign = tl.where(x < 0, -1.0, 1.0)
    x_abs = tl.abs(x)

    use_identity = x_abs > 1.0
    y = tl.where(use_identity, 1.0 / x_abs, x_abs)

    #  p3*y^3 + p1*y, close atan
    p1 = 0.9998660
    p3 = -0.288679       # p5 = 0.1801410
    atan_y = y * (p1 + y * y * p3)    # atan_y = y * (p1 + y2 * (p3 + y2 * p5))
    PI_HALF = 3.141592653589793 / 2.0
    
    result = tl.where(use_identity, PI_HALF - atan_y, atan_y)
    result = result * sign
    tl.store(output_ptr + offsets, result, mask=mask)

def triton_atan(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    atan_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
