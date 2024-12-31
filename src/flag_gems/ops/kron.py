import torch
import triton
import triton.language as tl

from .. import runtime
from ..utils import triton_lang_extension as tle


def prepare_tensor_for_kron(tensor_a, tensor_b):
    if tensor_a.numel() == 0 or tensor_b.numel() == 0:
        return tensor_a, tensor_b, ()

    if tensor_a.dim() == 0:
        tensor_a = tensor_a.unsqueeze(0)
    if tensor_b.dim() == 0:
        tensor_b = tensor_b.unsqueeze(0)

    a_shape = (
        list(tensor_a.shape)
        if len(tensor_a.shape) >= 2
        else [1] * (2 - len(tensor_a.shape)) + list(tensor_a.shape)
    )
    b_shape = (
        list(tensor_b.shape)
        if len(tensor_b.shape) >= 2
        else [1] * (2 - len(tensor_b.shape)) + list(tensor_b.shape)
    )

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


@triton.autotune(configs=runtime.get_tuned_config("kron"), key=["M", "N"])
@triton.jit
def kron_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    M1,
    M2,
    N1,
    N2,
    a_stride_0,
    a_stride_1,
    b_stride_0,
    b_stride_1,
    c_stride_0,
    c_stride_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_an = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_am[:, None] < M) & (offs_an[None, :] < N)

    a_row = offs_am[:, None] // M2
    a_col = offs_an[None, :] // N2
    b_row = offs_am[:, None] % M2
    b_col = offs_an[None, :] % N2

    a_idx = a_row * a_stride_0 + a_col * a_stride_1
    b_idx = b_row * b_stride_0 + b_col * b_stride_1

    a = tl.load(a_ptr + a_idx, mask=mask)
    b = tl.load(b_ptr + b_idx, mask=mask)

    c = a * b

    c_idx = offs_am[:, None] * c_stride_0 + offs_an[None, :] * c_stride_1
    tl.store(c_ptr + c_idx, c, mask=mask)


def kron(A, B):
    if A.numel() == 0 or B.numel() == 0:
        return torch.empty(0, device=A.device, dtype=A.dtype)

    if A.dim() == 0 and B.dim() == 0:
        return A * B

    A_prepared, B_prepared, out_shape = prepare_tensor_for_kron(A, B)

    if not out_shape:
        return torch.empty(0, device=A.device, dtype=A.dtype)

    M1, N1 = A_prepared.shape[-2:]
    M2, N2 = B_prepared.shape[-2:]
    M, N = M1 * M2, N1 * N2

    batch_size = int(torch.prod(torch.tensor(out_shape[:-2]))) if out_shape[:-2] else 1

    C = torch.empty(out_shape, device=A.device, dtype=A.dtype)
    C_reshaped = C.view(-1, M, N)

    A_view = A_prepared.reshape(-1, M1, N1)
    B_view = B_prepared.reshape(-1, M2, N2)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    if not A_view.is_contiguous():
        A_view = A_view.contiguous()
    if not B_view.is_contiguous():
        B_view = B_view.contiguous()

    for i in range(batch_size):
        a_idx, b_idx = calculate_batch_indices(i, A_prepared.shape, B_prepared.shape)
        kron_kernel[grid](
            A_view[a_idx],
            B_view[b_idx],
            C_reshaped[i],
            M,
            N,
            M1,
            M2,
            N1,
            N2,
            A_view[a_idx].stride(0),
            A_view[a_idx].stride(1),
            B_view[b_idx].stride(0),
            B_view[b_idx].stride(1),
            C_reshaped[i].stride(0),
            C_reshaped[i].stride(1),
        )

    if A.dim() <= 1 and B.dim() <= 1 and A.numel() > 0 and B.numel() > 0:
        return C.reshape(-1)

    return C
