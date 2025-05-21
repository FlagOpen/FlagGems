import torch
import triton
import triton.language as tl

@triton.jit
def glu_kernel(
    a_ptr, b_ptr, output_ptr,
    stride_batch, stride_dim,
    size_batch, size_dim,
    BLOCK_SIZE: tl.constexpr
):
    batch_id = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < size_dim

    # 计算偏移：base_ptr + batch_id * stride_batch + offset * stride_dim
    a_offset = batch_id * stride_batch + offset * stride_dim
    b_offset = batch_id * stride_batch + offset * stride_dim
    out_offset = batch_id * stride_batch + offset * stride_dim

    a = tl.load(a_ptr + a_offset, mask=mask)
    b = tl.load(b_ptr + b_offset, mask=mask)

    b_float = b.to(tl.float32)
    sigmoid_b = 1 / (1 + tl.exp(-b_float))
    result = a * sigmoid_b

    tl.store(output_ptr + out_offset, result, mask=mask)


def glu(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    
    dim = dim if dim >= 0 else input_tensor.ndim + dim
    assert input_tensor.shape[dim] % 2 == 0, "Split dimension must be even"

    # Split into a and b
    a, b = torch.chunk(input_tensor, 2, dim=dim)
    output = torch.empty_like(a)

    # Flatten batch and compute strides
    batch_shape = list(a.shape)
    size_dim = batch_shape[dim]
    batch_shape[dim] = 1  # remove GLU dim
    size_batch = int(torch.prod(torch.tensor(batch_shape)))

    a_contig = a.contiguous()
    b_contig = b.contiguous()
    output_contig = output.contiguous()

    # Flatten for kernel use
    a_flat = a_contig.view(size_batch, size_dim)
    b_flat = b_contig.view(size_batch, size_dim)
    output_flat = output_contig.view(size_batch, size_dim)

    # Compute strides for flattened tensors
    stride_batch = a_flat.stride(0)
    stride_dim = a_flat.stride(1)

    BLOCK_SIZE = 1024
    grid = (size_batch,)

    glu_kernel[grid](
        a_flat, b_flat, output_flat,
        stride_batch, stride_dim,
        size_batch, size_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output
