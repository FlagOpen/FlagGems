import cmath
import logging

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def bit_reversal_indices_kernel(
    output_ptr,
    n_elements,
    num_bits,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    indices = offsets
    rev = tl.zeros_like(offsets)

    for j in range(num_bits):
        rev = tl.where(offsets & (1 << j), rev | (1 << (num_bits - 1 - j)), rev)
    indices = rev

    tl.store(output_ptr + offsets, indices, mask=mask)


@triton.jit
def fft_kernel(
    xr_ptr,
    xi_ptr,
    step,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    i = offsets
    starts = offsets
    j = offsets
    k = offsets

    starts = (offsets // step) * 2 * step
    i = offsets % step
    j = starts + i
    k = step + j
    theta = -cmath.pi * i / step

    xkr = tl.load(xr_ptr + k, mask=mask)
    xki = tl.load(xi_ptr + k, mask=mask)
    xjr = tl.load(xr_ptr + j, mask=mask)
    xji = tl.load(xi_ptr + j, mask=mask)

    t_real = tl.cos(theta) * xkr - tl.sin(theta) * xki
    t_imag = tl.sin(theta) * xkr + tl.cos(theta) * xki

    xkr = xjr - t_real
    xki = xji - t_imag
    xjr = xjr + t_real
    xji = xji + t_imag

    tl.store(xr_ptr + k, xkr, mask=mask)
    tl.store(xi_ptr + k, xki, mask=mask)

    tl.store(xr_ptr + j, xjr, mask=mask)
    tl.store(xi_ptr + j, xji, mask=mask)


def bit_reversal_indices(x: torch.Tensor, n):
    num_bits = int(np.log2(n))
    indices = torch.arange(n, device=x.device)
    assert x.is_cuda and indices.is_cuda
    n_elements = indices.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    bit_reversal_indices_kernel[grid](indices, n_elements, num_bits, BLOCK_SIZE=1024)
    return indices


# TODO(Bowen12992): support more cases
def rad2_fft(x: torch.Tensor, n=None, dim=-1, norm=None):
    logging.debug("GEMS FFT")
    n_elements = x.numel()
    assert n is None, "Not support assign `signal length` currently."
    assert norm is None, "Not support assign `normalization mode` currently."

    indices = bit_reversal_indices(x, len(x))

    if x.dtype is not torch.cfloat:
        xr = x[indices]
        xi = torch.zeros_like(x)
    else:
        xr = x.real[indices]
        xi = x.imag[indices]

    assert xi.is_cuda and xr.is_cuda
    grid = lambda meta: (triton.cdiv(n_elements // 2, meta["BLOCK_SIZE"]),)

    step = 1
    while step < n_elements:
        fft_kernel[grid](xr, xi, step, n_elements // 2, BLOCK_SIZE=1024)
        step *= 2

    output = torch.complex(xr, xi)
    return output
