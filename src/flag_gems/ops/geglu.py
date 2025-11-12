import logging
import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
pow = tl_extra_shim.pow
tanh = tl_extra_shim.tanh


logger = logging.getLogger(__name__)


import logging
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

erf = tl_extra_shim.erf
exp = tl_extra_shim.exp
pow = tl_extra_shim.pow
tanh = tl_extra_shim.tanh

logger = logging.getLogger(__name__)

@triton.jit
def geglu_kernel(input_ptr, output_ptr, M, H, stride_in_m, stride_in_h, stride_out_m, stride_out_h,
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_H: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)

    # input 切分为 x_a, x_b
    input_a_ptr = input_ptr + offs_m[:, None] * stride_in_m + offs_h[None, :] * stride_in_h
    input_b_ptr = input_ptr + offs_m[:, None] * stride_in_m + (offs_h[None, :] + H) * stride_in_h
    output_ptr = output_ptr + offs_m[:, None] * stride_out_m + offs_h[None, :] * stride_out_h

    x_a = tl.load(input_a_ptr, mask=mask, other=0.0).to(tl.float32)
    x_b = tl.load(input_b_ptr, mask=mask, other=0.0).to(tl.float32)

    gelu_out = 0.5 * x_a * (1 + tanh(0.79788456 * x_a * (1 + 0.044715 * pow(x_a, 2))))
    out = gelu_out * x_b

    tl.store(output_ptr, out.to(tl.float32), mask=mask)

def geglu(input_tensor: torch.Tensor) -> torch.Tensor:
    shape = input_tensor.shape
    H = shape[-1] // 2
    M = input_tensor.numel() // (2 * H)

    input_2d = input_tensor.contiguous().view(M, 2*H)
    output_2d = torch.empty(M, H, device=input_tensor.device, dtype=input_tensor.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(H, META['BLOCK_SIZE_H']),
    )

    geglu_kernel[grid](
        input_2d, output_2d,
        M, H,
        input_2d.stride(0), input_2d.stride(1),
        output_2d.stride(0), output_2d.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_H=64,
    )

    return output_2d.view(*shape[:-1], H)


