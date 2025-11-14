import torch
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from flag_gems.fused.silu_and_mul import silu_and_mul_kernel, silu_and_mul_grad_kernel
except ImportError:
    import sys
    import os
    flag_gems_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(flag_gems_src)
    from flag_gems.fused.silu_and_mul import silu_and_mul_kernel, silu_and_mul_grad_kernel


class _SwiGLUAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        quantizer: Optional[Any] = None  
    ) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.quantizer = quantizer  
        
       
        hidden_dim = input_tensor.shape[-1] // 2
        A = input_tensor[..., :hidden_dim].contiguous()
        B = input_tensor[..., hidden_dim:].contiguous()
        return silu_and_mul_kernel(A, B)
    
    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], None]:
        input_tensor, = ctx.saved_tensors
        
      
        hidden_dim = input_tensor.shape[-1] // 2
        A = input_tensor[..., :hidden_dim].contiguous()
        B = input_tensor[..., hidden_dim:].contiguous()
        grad_A, grad_B = silu_and_mul_grad_kernel(A, B, grad_output.contiguous())
        grad_input = torch.cat([grad_A, grad_B], dim=-1).contiguous()
        
        return grad_input, None


def swiglu(
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None  
) -> torch.Tensor:

    if input_tensor.shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension of input must be even (got {input_tensor.shape[-1]})")
    if not input_tensor.is_cuda:
        raise ValueError("Input tensor must be CUDA tensor")
    
    return _SwiGLUAutograd.apply(input_tensor.contiguous(), quantizer)


def dswiglu(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    quantizer: Optional[Any] = None  
) -> torch.Tensor:

    input_shape = input_tensor.shape
    grad_out_shape = grad_output.shape
    if input_shape[:-1] != grad_out_shape[:-1] or input_shape[-1] != 2 * grad_out_shape[-1]:
        raise ValueError(
            f"Shape mismatch: input {input_shape} vs grad_output {grad_out_shape} "
            f"(input last dim must be 2x grad_output last dim)"
        )
    if not all(t.is_cuda for t in [grad_output, input_tensor]):
        raise ValueError("grad_output and input_tensor must be CUDA tensors")
    
    hidden_dim = input_tensor.shape[-1] // 2
    A = input_tensor[..., :hidden_dim].contiguous()
    B = input_tensor[..., hidden_dim:].contiguous()
    grad_A, grad_B = silu_and_mul_grad_kernel(A, B, grad_output.contiguous())
    
    return torch.cat([grad_A, grad_B], dim=-1).contiguous().view(input_shape)

class SwiGLU:
    def __init__(
        self,
        *,
        cache_quantized_input: bool = False,
        quantizer: Optional[Any] = None  
    ):
        self.cache_quantized_input = cache_quantized_input
        self.quantizer = quantizer

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return swiglu(input_tensor, quantizer=self.quantizer)

__all__ = ["SwiGLU", "swiglu", "dswiglu"]