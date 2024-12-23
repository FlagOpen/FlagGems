import torch
import triton
import triton.language as tl

def prepare_tensor_for_kron(tensor_a, tensor_b):
    
    if tensor_a.numel() == 0 or tensor_b.numel() == 0:
        return tensor_a, tensor_b, ()

    if tensor_a.dim() == 0:
        tensor_a = tensor_a.unsqueeze(0)
    if tensor_b.dim() == 0:
        tensor_b = tensor_b.unsqueeze(0)

    a_shape = list(tensor_a.shape) if len(tensor_a.shape) >= 2 else [1] * (2 - len(tensor_a.shape)) + list(tensor_a.shape)
    b_shape = list(tensor_b.shape) if len(tensor_b.shape) >= 2 else [1] * (2 - len(tensor_b.shape)) + list(tensor_b.shape)

    if len(a_shape) > len(b_shape):
        b_shape = [1] * (len(a_shape) - len(b_shape)) + b_shape
    elif len(b_shape) > len(a_shape):
        a_shape = [1] * (len(b_shape) - len(a_shape)) + a_shape

    out_shape = tuple(a * b for a, b in zip(a_shape, b_shape))
    return tensor_a.reshape(*a_shape), tensor_b.reshape(*b_shape), out_shape

def calculate_batch_indices(i, shape_a, shape_b):
    
    a_batch_dims = shape_a[:-2] or (1,)
    b_batch_dims = shape_b[:-2] or (1,)
    out_batch_dims = tuple(a * b for a, b in zip(a_batch_dims, b_batch_dims))

    out_indices = []
    remaining = i
    for dim_size in out_batch_dims[::-1]:
        out_indices.insert(0, remaining % dim_size)
        remaining //= dim_size

    a_idx = 0
    b_idx = 0
    for out_idx, (a_dim, b_dim) in zip(out_indices, zip(a_batch_dims, b_batch_dims)):
        a_idx = a_idx * a_dim + out_idx // b_dim
        b_idx = b_idx * b_dim + out_idx % b_dim

    return a_idx, b_idx

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'NUM_STAGES': 1}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_STAGES': 1}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_STAGES': 1}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_STAGES': 1}),
    ],
    key=['total_elements']
)

@triton.jit
def kron_kernel(
    a_ptr, b_ptr, c_ptr,
    M1, M2, N1, N2,
    a_stride_0, a_stride_1,
    b_stride_0, b_stride_1,
    c_stride_0, c_stride_1,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements

    out_row = idx // (N1 * N2)
    out_col = idx % (N1 * N2)

    a_row = out_row // M2
    a_col = out_col // N2
    b_row = out_row % M2
    b_col = out_col % N2

    a_idx = a_row * a_stride_0 + a_col * a_stride_1
    b_idx = b_row * b_stride_0 + b_col * b_stride_1

    a = tl.load(a_ptr + a_idx, mask=mask)
    b = tl.load(b_ptr + b_idx, mask=mask)
    c = a * b

    c_idx = out_row * c_stride_0 + out_col * c_stride_1
    tl.store(c_ptr + c_idx, c, mask=mask)

def kron(A, B):
    
    if A.numel() == 0 or B.numel() == 0:
        empty = torch.tensor([], device=A.device, dtype=A.dtype)
        return empty.reshape(())  

    if A.dim() == 0 and B.dim() == 0:
        return A * B  

    A_prepared, B_prepared, out_shape = prepare_tensor_for_kron(A, B)

    if not out_shape:
        return torch.tensor([], device=A.device, dtype=A.dtype).reshape(())

    M1, N1 = A_prepared.shape[-2:]
    M2, N2 = B_prepared.shape[-2:]

    batch_size = 1
    for dim in out_shape[:-2]:
        batch_size *= dim

    total_elements = M1 * M2 * N1 * N2

    C = torch.empty(out_shape, device=A.device, dtype=A.dtype)
    C_reshaped = C.view(batch_size, M1 * M2, N1 * N2)

    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)

    A_view = A_prepared.view(-1, M1, N1)
    B_view = B_prepared.view(-1, M2, N2)

    for i in range(batch_size):
        a_idx, b_idx = calculate_batch_indices(i, A_prepared.shape, B_prepared.shape)
        kron_kernel[grid](
            A_view[a_idx].contiguous(),
            B_view[b_idx].contiguous(),
            C_reshaped[i],
            M1, M2, N1, N2,
            A_view[a_idx].stride(0), A_view[a_idx].stride(1),
            B_view[b_idx].stride(0), B_view[b_idx].stride(1),
            C_reshaped[i].stride(0), C_reshaped[i].stride(1),
            total_elements
        )

    if A.dim() <= 1 and B.dim() <= 1 and A.numel() > 0 and B.numel() > 0:
        return C.reshape(-1)

    return C
