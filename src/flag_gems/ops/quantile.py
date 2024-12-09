import logging

import torch
import triton
import triton.language as tl
from torch import Tensor, tensor

from ..utils import dim_compress, libentry

INTERPOLATION_METHOD = ["linear", "lower", "higher", "nearest", "midpoint"]


def cfggen(one_dim=False):
    block_q = tensor([1, 2, 4, 8], dtype=torch.int32)
    if one_dim:
        configs = [triton.Config({"BLOCK_Q": q.item()}, num_warps=4) for q in block_q]
    else:
        block_n = tensor([2**i for i in range(6, 11)], dtype=torch.int32)
        x, y = torch.meshgrid(block_n, block_q, indexing="ij")
        configs = [
            triton.Config({"BLOCK_Q": q.item(), "BLOCK_N": n.item()}, num_warps=4)
            for n, q in zip(x.ravel(), y.ravel())
        ]
    return configs

 
@libentry()
@triton.autotune(configs=cfggen(True), key=["M", "Q"])
@triton.jit
def quantile_kernel_1d(
    inp, q, out, M, Q, BLOCK_Q: tl.constexpr, interpolation: tl.constexpr
):
    pid = tl.program_id(0)
    ctype = inp.dtype.element_ty

    offsets = pid * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask = offsets < Q
    q_ptrs = q + offsets
    out_ptrs = out + offsets

    q_block = tl.load(q_ptrs, mask, 0.0).to(ctype) * (M - 1)
    q_lower = tl.floor(q_block).to(tl.int32)
    q_upper = tl.ceil(q_block).to(tl.int32)
    inp_lower = tl.load(inp + q_lower)
    inp_upper = tl.load(inp + q_upper)

    if interpolation == "linear":
        q_frac = q_block - q_lower
        tl.store(out_ptrs, inp_lower + (inp_upper - inp_lower) * q_frac, mask)

    elif interpolation == "lower":
        tl.store(out_ptrs, inp_lower, mask)

    elif interpolation == "higher":
        tl.store(out_ptrs, inp_upper, mask)

    elif interpolation == "nearest":
        q_near = tl.where(q_block - q_lower > q_upper - q_block, inp_upper, inp_lower)
        tl.store(out_ptrs, q_near, mask)

    elif interpolation == "midpoint":
        tl.store(out_ptrs, (inp_lower + inp_upper) / 2, mask)


def quantile(inp, q, *, interpolation="linear", out=None) -> Tensor:
    logging.debug("GEMS QUANTILE")
    assert torch.is_floating_point(inp)
    assert isinstance(q, (float, torch.Tensor))
    assert interpolation in INTERPOLATION_METHOD

    M = inp.numel()
    if isinstance(q, float):
        q = torch.tensor(q, device=inp.device)
    Q = len(q)

    assert M > 0
    assert Q > 0
    assert torch.all(q >= 0.0) and torch.all(q <= 1.0)

    inp, _ = inp.sort()  # Sort the input with torch.sort()
    output = torch.empty(q.shape, dtype=inp.dtype, device=inp.device)
    grid = lambda meta: [triton.cdiv(Q, meta["BLOCK_Q"])]

    with torch.cuda.device(inp.device):
        quantile_kernel_1d[grid](inp, q, output, M, Q, interpolation=interpolation)

    if out is not None:
        out.copy_(output)
    return output


@libentry()
@triton.autotune(configs=cfggen(), key=["N", "M", "Q"])
@triton.jit
def quantile_kernel_2d(
    inp,
    q,
    out,
    N,
    M,
    Q,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    interpolation: tl.constexpr,
):
    pid_Q = tl.program_id(0)
    pid_N = tl.program_id(1)
    ctype = inp.dtype.element_ty

    offsets_Q = pid_Q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask_Q = offsets_Q < Q
    q_ptrs = q + offsets_Q

    offsets_N = pid_N * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_N = offsets_N < N

    out_ptrs = out + offsets_N[:, None] * Q + offsets_Q[None, :]
    mask_out = mask_N[:, None] & mask_Q[None, :]

    q_block = tl.load(q_ptrs, mask_Q, 0.0).to(ctype) * (M - 1)
    q_lower = tl.floor(q_block).to(tl.int32)
    q_upper = tl.ceil(q_block).to(tl.int32)

    inp_lower = tl.load(inp + offsets_N[:, None] * M + q_lower[None, :])
    inp_upper = tl.load(inp + offsets_N[:, None] * M + q_upper[None, :])

    if interpolation == "linear":
        q_frac = q_block - q_lower
        tl.store(out_ptrs, inp_lower + (inp_upper - inp_lower) * q_frac, mask_out)

    elif interpolation == "lower":
        tl.store(out_ptrs, inp_lower, mask_out)

    elif interpolation == "higher":
        tl.store(out_ptrs, inp_upper, mask_out)

    elif interpolation == "nearest":
        q_near = tl.where(q_block - q_lower > q_upper - q_block, inp_upper, inp_lower)
        tl.store(out_ptrs, q_near, mask_out)

    elif interpolation == "midpoint":
        tl.store(out_ptrs, (inp_lower + inp_upper) / 2, mask_out)


def quantile_dim(inp, q, dim=None, keepdim=False, *, interpolation="linear", out=None) -> Tensor:
    logging.debug("GEMS QUANTILE DIM")
    assert torch.is_floating_point(inp)
    assert dim is None or isinstance(dim, int)
    assert isinstance(q, (float, torch.Tensor))
    assert interpolation in INTERPOLATION_METHOD

    M = inp.numel()
    if isinstance(q, float):
        q = torch.tensor(q, device=inp.device)
    Q = len(q)

    assert M > 0
    assert Q > 0
    assert torch.all(q >= 0.0) and torch.all(q <= 1.0)

    if dim is None:
        inp = inp.ravel()
        dim = 0

    shape = list(inp.shape)

    dim %= inp.ndim
    inp = dim_compress(inp, dim)
    M = shape[dim]
    N = inp.numel() // M

    inp, _ = inp.sort()  # Sort the input with torch.sort()
    output = torch.empty(inp.shape[:-1] + (Q,), dtype=inp.dtype, device=inp.device)

    grid = lambda meta: [
        triton.cdiv(Q, meta["BLOCK_Q"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    ]

    with torch.cuda.device(inp.device):
        quantile_kernel_2d[grid](inp, q, output, N, M, Q, interpolation=interpolation)

    output = output.permute(
        (-1,) + tuple(range(0, inp.ndim - 1))
    )  # Same as torch.quantile()
    if keepdim:
        output = output.unsqueeze(dim + 1)
        
    if out is not None:
        out.copy_(output)
    return output
