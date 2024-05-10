import torch
import triton
import triton.language as tl
from .__libentry__ import libentry


def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4) for m in block_m
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def var_mean_kernel(
    X,
    Var,
    Mean,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X it should compute.
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Var = Var + pid
    Mean = Mean + pid
    row_mask = pid < M

    # Compute mean
    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    _mean /= N
    mean = tl.sum(_mean, axis=1)[:, None]

    # Compute variance
    _var = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        _var += x * x
    _var /= N - correction
    var = tl.sum(_var, axis=1)[:, None]
    # Write mean / var
    tl.store(Mean, mean, row_mask)
    tl.store(Var, var, row_mask)


def var_mean(x, dim=None, *, correction=None, keepdim=False):
    if __debug__:
        print("GEMS VAR MEAN")
    if correction is None:
        correction = 1.0

    if dim is None:
        dim = 0
        M = 1
        N = x.numel()
        shape = [1] * x.ndim
        x = x.flatten()
    else:
        dim = dim[0] % x.ndim
        x = x.transpose(dim, -1)
        N = x.shape[-1]
        M = x.numel() // N
        shape = list(x.shape)
        shape[-1] = 1
    x = x.contiguous()
    var = torch.empty(shape, dtype=x.dtype, device=x.device)
    mean = torch.empty(shape, dtype=x.dtype, device=x.device)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    var_mean_kernel[grid](x, var, mean, M, N, correction)
    var = var.transpose(dim, -1)
    mean = mean.transpose(dim, -1)
    if not keepdim:
        var = var.squeeze()
        mean = mean.squeeze()
    return var, mean
