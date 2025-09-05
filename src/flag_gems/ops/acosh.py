import logging
import triton
import triton.language as tl
import torch

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

@triton.jit
def acosh_forward_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载输入数据
    x = tl.load(input_ptr + offsets, mask=mask)

    # step1: 若x < 1, 则返回 nan
    one = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    # step2: 计算(x^2 - 1)
    x_sq_minus_one = tl.where(x >= 1, x * x - one, tl.full((BLOCK_SIZE,), float('nan'), dtype=tl.float32))
    # step3: 计算sqrt(x^2 - 1)
    sqrt_val = tl.sqrt(tl.maximum(x_sq_minus_one, 0.0))
    # step4: 计算x + sqrt(x^2 - 1)
    sum_val = x + sqrt_val
    # step5: 计算ln(x + sqrt(x^2 - 1))
    result = tl.log(sum_val)
    # step6: 保存结果
    tl.store(output_ptr + offsets, result, mask=mask)

def acosh(input: torch.Tensor):
    output = torch.empty_like(input)

    n_elements = input.numel()

    BLOCK_SIZE = 1024

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    acosh_forward_kernel[grid](
        input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

    return output