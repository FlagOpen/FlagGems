import torch
import triton
import triton.language as tl


def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 128}, num_warps=4) for m in block_m
    ]
    return configs


@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def add_on_kernel(
    idx,
    add_on,
    cur_shape,
    cur_strides,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows_offset = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offset < M

    for off in range(0, N, BLOCK_N):
        cols_offset = off + tl.arange(0, BLOCK_N)[None, :]
        cols_mask = cols_offset < N
        block_mask = rows_mask and cols_mask

        offsets = rows_offset * N + cols_offset
        cur_idx = tl.load(idx + offsets, mask=block_mask, other=0)
        mod = (cur_idx % cur_shape).to(tl.int32)
        res = mod * cur_strides
        tl.store(add_on + offsets, res, mask=block_mask)


def offset_calculator(inp, idx, strides, dim, isInp):
    ndim = inp.ndim
    shape = list(inp.shape)
    offsets = torch.zeros_like(inp, dtype=torch.int32, device=inp.device)
    idx_dim = torch.zeros_like(inp, dtype=torch.int32, device=inp.device)
    add_on = torch.zeros_like(inp, dtype=torch.int32, device=inp.device)

    for d in range(0, ndim):
        N = inp.size(inp.ndim - 1)
        M = inp.numel() // N
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        add_on_kernel[grid](idx, add_on, shape[d], strides[d], M, N)

        offsets = torch.add(offsets, add_on)
        if d == dim:
            idx_dim = add_on
        idx = triton.cdiv(idx, shape[d])
    return offsets if not isInp else (offsets - idx_dim)
