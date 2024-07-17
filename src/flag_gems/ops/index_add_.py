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


def cfggen():
    block_m = [1]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 4}, num_warps=1) for m in block_m
    ]
    return configs


@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def index_add_kernel(
    inp, index, src, M, N, alpha, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    rows_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M

    for off in range(0, N, BLOCK_N):
        cols_offsets = off + tl.arange(0, BLOCK_N)
        cols_mask = cols_offsets < N
        block_mask = rows_mask and cols_mask

        cur_indices = tl.load(index + cols_offsets, mask=cols_mask, other=0)
        inp_off = rows_offsets * N + cur_indices[None, :]
        cur_inp = tl.load(inp + inp_off, mask=block_mask, other=0)
        src_off = rows_offsets * N + cols_offsets[None, :]
        cur_src = tl.load(src + src_off, mask=block_mask, other=0)

        cur_inp += alpha * cur_src

        tl.store(inp + inp_off, cur_inp, mask=block_mask)


def index_add_(inp, dim, index, src, alpha=1):
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device="cuda")
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert (
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "index.size(d) == self.size(d) for all dimensions d != dim"

    dim = dim % inp.ndim
    src_shape = list(src.shape)
    inp = dim_compress(inp, dim)
    src = dim_compress(src, dim)
    N = src_shape[dim]
    M = src.numel() // N

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    index_add_kernel[grid](inp, index, src, M, N, alpha)
    return inp


x1 = torch.ones(3, 3, device="cuda")
x2 = torch.ones(3, 3, device="cuda")
src = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float, device="cuda")
index = torch.tensor([0, 1, 2], device="cuda")
dim = 0
alpha = 1
triton_res = index_add_(x1, dim, index, src, alpha)
torch_res = torch.index_add(x2, dim, index, src, alpha=alpha)
print("triton_res", triton_res)
print("torch_res", torch_res)
