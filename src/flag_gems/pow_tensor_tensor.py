import torch
import triton
import triton.language as tl
from .__libentry__ import libentry


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"M_BLOCK_SIZE": 256}, num_warps=2, num_stages=4),
        triton.Config({"M_BLOCK_SIZE": 256}, num_warps=2, num_stages=5),
        triton.Config({"M_BLOCK_SIZE": 512}, num_warps=2, num_stages=4),
        triton.Config({"M_BLOCK_SIZE": 512}, num_warps=2, num_stages=5),
        triton.Config({"M_BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"M_BLOCK_SIZE": 1024}, num_warps=4, num_stages=5),
        triton.Config({"M_BLOCK_SIZE": 2048}, num_warps=4, num_stages=4),
        triton.Config({"M_BLOCK_SIZE": 2048}, num_warps=4, num_stages=5),
    ],
    key=["M"],
)
@triton.jit
def pow_tensor_tensor_kernel(
    X,
    exponent,
    Y,
    M,
    M_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) * M_BLOCK_SIZE
    Y_ptrs = tl.make_block_ptr(
        Y,
        shape=(M,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(M_BLOCK_SIZE,),
        order=(0,),
    )
    X_ptrs = tl.make_block_ptr(
        X,
        shape=(M,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(M_BLOCK_SIZE,),
        order=(0,),
    )
    exp_ptrs = tl.make_block_ptr(
        exponent,
        shape=(M,),
        strides=(1,),
        offsets=(pid,),
        block_shape=(M_BLOCK_SIZE,),
        order=(0,),
    )
    X_val = tl.load(X_ptrs)
    exp_val = tl.load(exp_ptrs)
    Y_val = tl.math.pow(X_val.to(tl.float32), exp_val)
    tl.store(Y_ptrs, Y_val.to(X_val.dtype))


def pow_tensor_tensor(A, exponent):
    if __debug__:
        print("GEMS POW_TENSOR_TENSOR")
    try:
        A, exponent = torch.broadcast_tensors(A, exponent)
    except RuntimeError as e:
        print(
            f"Pow: Tensor shape {A.shape} and tensor shape {exponent.shape} cannot broadcast to each other."
        )
    A = A.contiguous()
    exponent = exponent.contiguous()
    O = torch.empty_like(A)
    M = A.numel()
    grid_fn = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
    pow_tensor_tensor_kernel[grid_fn](A, exponent, O, M)
    return O
