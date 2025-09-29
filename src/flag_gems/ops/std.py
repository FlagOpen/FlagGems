import torch
import triton
import triton.language as tl

@triton.jit
def std_kernel_fused(
    input_ptr,
    output_ptr,  
    n_elements,
    unbiased: tl.constexpr,  
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    partial_sum = tl.sum(x, axis=0)
    partial_sum_sq = tl.sum(x * x, axis=0)
    
    partial_sum = partial_sum.to(tl.float32) 
    partial_sum_sq = partial_sum_sq.to(tl.float32)
    
    sum_total = tl.zeros((1,), dtype=tl.float32)
    sum_sq_total = tl.zeros((1,), dtype=tl.float32)
    
    tl.atomic_add(sum_total, partial_sum)
    tl.atomic_add(sum_sq_total, partial_sum_sq)
    
    if pid == 0:
        mean_val = sum_total[0] / n_elements
        # E[X^2] - (E[X])^2
        variance = (sum_sq_total[0] / n_elements) - (mean_val * mean_val)
        
        if unbiased and n_elements > 1:
            variance = variance * (n_elements / (n_elements - 1))
        
        std_dev = tl.sqrt(variance) 
        tl.store(output_ptr, std_dev)

def triton_std_fused(input: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    n_elements = input.numel()
    if n_elements == 0:
        return torch.tensor(0.0, device=input.device)
    
    std_result = torch.zeros(1, device=input.device, dtype=torch.float32)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    std_kernel_fused[grid](input, std_result, n_elements, unbiased, BLOCK_SIZE=BLOCK_SIZE)
    return std_result


