import torch
import triton
import triton.language as tl


def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()


def expand_index(index, inp_shape, dim):
    index_ = []
    # TODO
    return tuple(index_)


def cfggen():
    block_m = [4]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 2}, num_warps=1) for m in block_m
    ]
    return configs


@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def index_select_kernel(
    inp, out, M, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    rows_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M

    for off in range(0, index_len, BLOCK_N):
        cols_offsets = off + tl.arange(0, BLOCK_N)
        block_mask = rows_mask and (cols_offsets < N)
        out_mask = rows_mask and (cols_offsets < index_len)

        indices = tl.load(
            index + cols_offsets, mask=(cols_offsets < index_len), other=0
        )
        inp_off = rows_offsets * N + indices[None, :]
        out_off = rows_offsets * index_len + cols_offsets[None, :]

        selected = tl.load(inp + inp_off, mask=block_mask, other=0.0)
        tl.store(out + out_off, selected, mask=out_mask)


def index_select(inp, dim, index):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    assert ((i >= 0 and i < inp.size(dim)) for i in index), "Index out of range"

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    inp_shape = list(inp.shape)

    """
    inp = dim_compress(inp, dim)
    #print("inp_reshape", inp)
    N = inp_shape[dim]
    M = inp.numel() // N
    index_len = index.numel()
    out_shape = list(inp.shape)
    out_shape[inp.ndim - 1] = index.numel()
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    index_select_kernel[grid](inp, out, M, N, index, index_len)
    res_shape = inp_shape
    res_shape[dim] = index.numel()
    """
    print(expand_index(index, inp_shape, dim))

    return torch.gather(inp, dim, index)


inp = torch.arange(1, 61, device="cuda").reshape(3, 4, 5)
print("inp", inp)
indices = torch.tensor([0, 2], device="cuda")
dim = 1
torch_res = torch.index_select(inp, dim, indices)
print("torch_res", torch_res)
triton_res = index_select(inp, dim, indices)
print("triton_res", triton_res)
print("✅" if torch.allclose(triton_res, torch_res) else "❌")
