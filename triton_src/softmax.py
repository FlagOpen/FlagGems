import triton
import triton.language as tl


@triton.jit
def softmax_kernel_inner(
    output_ptr, input_ptr,
    stride_output_m, stride_output_n,
    stride_input_m, stride_input_n,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    input_ptrs = input_ptr + pid * stride_input_m + offsets * stride_input_n
    output_ptrs = output_ptr + pid * stride_output_m + offsets * stride_output_n

    row = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(row, axis=0)
    row = row - row_max
    exp_row = tl.exp(row)
    denom = tl.sum(exp_row, axis=0) + 1e-10
    softmax = exp_row / denom
    softmax = softmax.to(tl.float16)  
    tl.store(output_ptrs, softmax, mask=mask)


@triton.jit
def softmax_kernel_non_inner(
    output_ptr, input_ptr,
    stride_output_m, stride_output_n, stride_output_k,
    stride_input_m, stride_input_n, stride_input_k,
    M, N, K,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    if pid_m >= M or pid_k >= K:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    input_ptrs = input_ptr + pid_m * stride_input_m + offsets * stride_input_n + pid_k * stride_input_k
    output_ptrs = output_ptr + pid_m * stride_output_m + offsets * stride_output_n + pid_k * stride_output_k

    x = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0) + 1e-10
    softmax = (exp_x / denom).to(tl.float16)
    tl.store(output_ptrs, softmax, mask=mask)


@triton.jit
def softmax_backward_kernel_inner(
    output_ptr, grad_output_ptr, grad_input_ptr,
    stride_m, stride_n,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    output_ptrs = output_ptr + row * stride_m + cols * stride_n
    grad_out_ptrs = grad_output_ptr + row * stride_m + cols * stride_n
    grad_in_ptrs = grad_input_ptr + row * stride_m + cols * stride_n

    output = tl.load(output_ptrs, mask=mask, other=0.0)
    grad_output = tl.load(grad_out_ptrs, mask=mask, other=0.0)

    dot = tl.sum(output * grad_output, axis=0)
    grad_input = output * (grad_output - dot)

    tl.store(grad_in_ptrs, grad_input, mask=mask)

@triton.jit
def softmax_backward_kernel_non_inner(
    output_ptr, grad_output_ptr, grad_input_ptr,
    stride_m, stride_n, stride_k,
    M, N, K,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    offset = pid_m * stride_m + pid_k * stride_k + cols * stride_n
    output = tl.load(output_ptr + offset, mask=mask, other=0.0)
    grad_output = tl.load(grad_output_ptr + offset, mask=mask, other=0.0)

    dot = tl.sum(output * grad_output, axis=0)
    grad_input = output * (grad_output - dot)

    tl.store(grad_input_ptr + offset, grad_input, mask=mask)
