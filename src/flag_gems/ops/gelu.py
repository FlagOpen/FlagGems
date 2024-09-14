import logging
import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic

try:
    from triton.language.extra.cuda.libdevice import erf, pow, tanh
except ImportError:
    try:
        from triton.language.math import erf, pow, tanh
    except ImportError:
        from triton.language.libdevice import erf, pow, tanh


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_none(x):
    scale: tl.constexpr = 0.7071067811
    output = 0.5 * x * (1 + erf(x * scale))
    return output


@triton.jit
def gelu_forward_custom_kernel(
    x_ptr: tl.tensor, # *Pointer* to first input vector.
    output_ptr: tl.tensor, # *Pointer* to output vector.
    n_elements: int, # Size of the vector.
    BLOCK_SIZE: tl.constexpr, # Number of elements each program should process.
                            # NOTE: constexpr' so it can be used as a shape value.
):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0) # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)

    #No need to add offset and mask, as its stride is 0
    scale: tl.constexpr = 0.7071067811
    output = 0.5 * x * (1 + erf(x * scale))

    # Write output back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

def gelu_forward_custom(x: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output. numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta[ 'BLOCK_SIZE' ]),)
    gelu_forward_custom_kernel[grid](x, output, n_elements, BLOCK_SIZE=2048)
    return output


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def gelu_tanh(x):
    output = (
        0.5 * x * (1 + tanh(x * 0.79788456 * (1 + 0.044715 * pow(x.to(tl.float32), 2))))
    )
    return output


def gelu(A, *, approximate="none"):
    logging.debug("GEMS GELU")
    if approximate == "tanh":
        return gelu_tanh(A)
    else:
        if A.numel() % 1024:
            return gelu_forward_custom(A)
        else:
            return gelu_none(A)
