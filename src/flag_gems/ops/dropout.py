import triton
import triton.language as tl
import torch
import logging
from ..utils.random_utils import philox_cuda_seed_offset
from ..utils import libentry


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"N_BLOCK_SIZE": 256}, num_warps=2, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 256}, num_warps=2, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 512}, num_warps=2, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 512}, num_warps=2, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 1024}, num_warps=4, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 2048}, num_warps=4, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 2048}, num_warps=4, num_stages=5),
    ],
    key=[
        "N",
    ],
)
@triton.jit
def dropout_forward_kernel(
    X,
    Y,
    N,
    p,
    philox_seed,
    philox_offset,
    N_BLOCK_SIZE: tl.constexpr,
):
    block_offset = tl.program_id(0) * N_BLOCK_SIZE
    X_ptr = tl.make_block_ptr(
        X,
        shape=(N,),
        strides=(1,),
        offsets=(block_offset,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    Y_ptr = tl.make_block_ptr(
        Y,
        shape=(N,),
        strides=(1,),
        offsets=(block_offset,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    inp = tl.load(X_ptr)
    offset = philox_offset + block_offset + tl.arange(0, N_BLOCK_SIZE)
    pmask = tl.rand(philox_seed, offset, n_rounds=6) > p
    p = 1.0 / (1.0 - p)
    out = tl.where(pmask, inp * p, 0.0)
    tl.store(Y_ptr, out.to(inp.dtype))


@libentry()
@triton.autotune(
    [
        triton.Config({"N_BLOCK_SIZE": 256}, num_warps=2, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 256}, num_warps=2, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 512}, num_warps=2, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 512}, num_warps=2, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 1024}, num_warps=4, num_stages=5),
        triton.Config({"N_BLOCK_SIZE": 2048}, num_warps=4, num_stages=4),
        triton.Config({"N_BLOCK_SIZE": 2048}, num_warps=4, num_stages=5),
    ],
    key=[
        "N",
    ],
)
@triton.jit
def dropout_backward_kernel(
    DY,
    MASK,
    DX,
    N,
    p,
    philox_seed,
    philox_offset,
    N_BLOCK_SIZE: tl.constexpr,
):
    block_offset = tl.program_id(0) * N_BLOCK_SIZE
    DY_ptr = tl.make_block_ptr(
        DY,
        shape=(N,),
        strides=(1,),
        offsets=(block_offset,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    DX_ptr = tl.make_block_ptr(
        DX,
        shape=(N,),
        strides=(1,),
        offsets=(block_offset,),
        block_shape=(N_BLOCK_SIZE,),
        order=(0,),
    )
    dy = tl.load(DY_ptr)
    offset = philox_offset + block_offset + tl.arange(0, N_BLOCK_SIZE)
    pmask = tl.rand(philox_seed, offset, n_rounds=6) > p
    p = 1.0 / (1.0 - p)
    dx = tl.where(pmask, dy * p, 0.0)
    tl.store(DX_ptr, dx.to(dy.dtype))


class NativeDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, train):
        logging.debug("GEMS NATIVE DROPOUT FORWARD")
        assert p > 0.0 and p < 1.0, "p must be in (0, 1)"
        x = x.contiguous()
        O = torch.empty_like(x)
        N = x.numel()
        grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
        inc = triton.cdiv(N, BLOCK_SIZE)
        philox_seed, philox_offset = philox_cuda_seed_offset(inc)
        dropout_forward_kernel[grid_fn](x, O, N, p, philox_seed, philox_offset)
        ctx.save_for_backward(Mask)
        ctx.p = p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        return O, Mask

    @staticmethod
    def backward(ctx, grad_outputs, kwargs):
        logging.debug("GEMS NATIVE DROPOUT BACKWARD")
        (Mask,) = ctx.saved_tensors
        scale = 1.0 / (1.0 - ctx.p)
        grad_outputs = grad_outputs.contiguous()
        grad_inputs = torch.empty_like(grad_outputs)
        N = grad_outputs.numel()
        grid_fn = lambda meta: (triton.cdiv(N, meta["N_BLOCK_SIZE"]),)
        dropout_backward_kernel[grid_fn](grad_outputs, grad_inputs, N, p, ctx.philox_seed, ctx.philox_offset)
        return grad_inputs, None, None


def native_dropout(x, p=0.5, train=True):
    return NativeDropout.apply(x, p, train)
