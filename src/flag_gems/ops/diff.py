import torch
import triton
import triton.language as tl
from torch import Tensor, tensor

from .. import runtime
from ..runtime import torch_device_fn
from ..utils import dim_compress, libentry, libtuner
from ..utils import triton_lang_extension as tle


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("diff_1d"),
    key=["N"],
)
@triton.jit
def diff_kernel_1d(in_ptr, out_ptr, N, N_bound, BLOCK_DIFF: tl.constexpr):
    pid = tle.program_id(0)

    in_offsets = pid * BLOCK_DIFF + tl.arange(0, BLOCK_DIFF)
    mask_in = in_offsets < N_bound - 1
    in_block = tl.load(in_ptr + in_offsets, mask_in)
    next_block = tl.load(in_ptr + in_offsets + 1, mask_in)
    tl.store(out_ptr + in_offsets, next_block - in_block, mask_in)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("diff"),
    key=["M", "N"],
)
@triton.jit
def diff_kernel_2d(
    in_ptr, out_ptr, M, N, N_bound, BLOCK_M: tl.constexpr, BLOCK_DIFF: tl.constexpr
):
    pid_M = tle.program_id(0)
    pid_diff = tle.program_id(1)

    M_offsets = pid_M * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_M = M_offsets < M

    in_offsets_diff = pid_diff * BLOCK_DIFF + tl.arange(0, BLOCK_DIFF)
    mask_in_diff = in_offsets_diff < N_bound - 1
    in_offsets = M_offsets[:, None] * N + in_offsets_diff
    mask_in = mask_M[:, None] & mask_in_diff
    out_offsets = M_offsets[:, None] * N + in_offsets_diff

    in_block = tl.load(in_ptr + in_offsets, mask_in)
    next_block = tl.load(in_ptr + in_offsets + 1, mask_in)
    tl.store(out_ptr + out_offsets, next_block - in_block, mask_in)


def diff(input, n=1, dim=-1, prepend=None, append=None) -> Tensor:
    if prepend is not None:
        input = torch.cat([prepend, input], dim=dim)
    if append is not None:
        input = torch.cat([input, append], dim=dim)

    if n <= 0:
        return input

    shape = list(input.shape)
    dim = dim % input.ndim
    reduce_len = shape[dim]

    if n >= reduce_len:
        empty_tensor = tensor([], dtype=input.dtype, device=input.device)
        return torch.reshape(empty_tensor, shape[:dim] + [0] + shape[(dim + 1) :])

    input = dim_compress(input, dim)
    N = reduce_len
    M = input.numel() // N

    output = torch.zeros(input.shape, device=input.device, dtype=input.dtype)

    n_steps = n
    while n_steps > 0:
        cur_in_diff_len = N - (n - n_steps)
        if len(shape) == 1:
            grid = lambda meta: (triton.cdiv(cur_in_diff_len, meta["BLOCK_DIFF"]),)
            with torch_device_fn.device(input.device):
                diff_kernel_1d[grid](input, output, N, cur_in_diff_len)
        else:
            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(cur_in_diff_len, meta["BLOCK_DIFF"]),
            )
            with torch_device_fn.device(input.device):
                diff_kernel_2d[grid](input, output, M, N, cur_in_diff_len)
        n_steps -= 1
        input.copy_(output)

    output = output[..., : (N - n)].contiguous()
    output = torch.moveaxis(output, -1, dim)
    return output
