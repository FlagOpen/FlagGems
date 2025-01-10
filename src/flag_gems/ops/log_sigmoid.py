import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=1, num_warps=4),
    ], key=['n_elements'])
@triton.jit
def log_sigmoid_forward(x_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    log2e: tl.constexpr = 1.4426950408889634
    threshold: tl.constexpr = -10.0  
    
    result = tl.where(x < threshold, x, -tl.log(1 + tl.exp2(-x.to(tl.float32) * log2e)))
    
    tl.store(output_ptr + offsets, result, mask=mask)


def log_sigmoid(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device='cuda')
    
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
  
    log_sigmoid_forward[grid](x_ptr=x, output_ptr=output, n_elements=n_elements)
    
    return output
