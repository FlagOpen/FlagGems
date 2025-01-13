import logging

import torch
import triton
import triton.language as tl
from torch import Tensor, tensor

from ..runtime import torch_device_fn
from ..utils import dim_compress, libentry
from ..utils import triton_lang_extension as tle


@libentry()
@triton.jit
def diff_kernel_1d(in_ptr, out_ptr, coeff_ptr, n: tl.constexpr):
    pid = tle.program_id(0)

    coeff_offsets = tl.arange(0, triton.next_power_of_2(n + 1))
    in_offsets = pid + coeff_offsets
    out_offset = pid

    mask_co_in = coeff_offsets < n + 1

    in_block = tl.load(in_ptr + in_offsets, mask_co_in)
    coeff = tl.load(coeff_ptr + coeff_offsets, mask_co_in)
    result = tl.sum(in_block * coeff)
    tl.store(out_ptr + out_offset, result)


@libentry()
@triton.jit
def diff_kernel_2d(
    in_ptr, out_ptr, coeff_ptr, M, N, n: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_diff = tle.program_id(1)
    pid_n = tle.program_id(0)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < M

    coeff_offsets = tl.arange(0, triton.next_power_of_2(n + 1))

    in_offsets_diff = pid_diff + coeff_offsets
    in_offsets = n_offsets[:, None] * N + in_offsets_diff

    out_offset_diff = pid_diff
    out_offsets = n_offsets * tle.num_programs(1) + out_offset_diff

    mask_co_in = coeff_offsets < n + 1
    mask_in = mask_n[:, None] & mask_co_in
    mask_out = mask_n

    in_block = tl.load(in_ptr + in_offsets, mask_in)
    coeff = tl.load(coeff_ptr + coeff_offsets, mask_co_in)
    result = tl.sum(in_block * coeff, axis=1)
    tl.store(out_ptr + out_offsets, result, mask_out)


def bin_coeff(n, device):
    # fg introduce errors
    # coeff = torch.ones(n + 1, dtype=torch.int64, device=device)
    coeff = [1] * (n + 1)
    # coeff[-1] = 1
    for i in range(1, n + 1):
        coeff[n - i] = coeff[n + 1 - i] * (n - i + 1) // i * (-1)
    return torch.tensor(coeff, dtype=torch.int64, device=device)


def diff(input, n=1, dim=-1, prepend=None, append=None) -> Tensor:
    if prepend is not None:
        input = torch.cat([prepend, input], dim=dim)
    if append is not None:
        input = torch.cat([input, append], dim=dim)

    if n == 0:
        return input

    max_len = input.shape[dim % input.ndim]
    if n >= max_len:
        logging.warning(
            "Cannot conduct diff(input, n) with diff length = {} and n = {}.".format(
                max_len, n
            )
        )
        return tensor([], dtype=input.dtype, device=input.device)

    if input.ndim == 1:
        shape = list(input.shape)
        dim = dim % input.ndim
        input = dim_compress(input, dim)

        output_diff_len = shape[dim] - n
        output = torch.zeros(output_diff_len, device=input.device, dtype=input.dtype)

        coeff = bin_coeff(n, input.device)

        grid = [output_diff_len]
        with torch_device_fn.device(input.device):
            diff_kernel_1d[grid](input, output, coeff, n)
        return output

    shape = list(input.shape)
    dim = dim % input.ndim
    input = dim_compress(input, dim)
    N = shape[dim]
    M = input.numel() // N

    output_diff_len = shape[dim] - n
    output_shape = shape[:dim] + shape[(dim + 1) :] + [output_diff_len]
    output = torch.zeros(output_shape, device=input.device, dtype=input.dtype)

    coeff = bin_coeff(n, input.device)

    block_size = 16
    grid = [triton.cdiv(M, block_size), output_diff_len]
    with torch_device_fn.device(input.device):
        diff_kernel_2d[grid](input, output, coeff, M, N, n, block_size)
    output = torch.moveaxis(output, -1, dim)
    return output
