import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry
from ..utils import triton_lang_extension as tle


@libentry()
@triton.jit
def linspace_kernel(
    out_ptr,
    out_stride0,
    start,
    mid,
    end,
    step_size,
    steps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < steps
    fw_mask = idx < mid
    fw_values = start + (step_size * idx)
    bd_values = end - step_size * (steps - idx - 1)

    out_val = tl.where(fw_mask, fw_values, bd_values)
    tl.store(out_ptr + idx * out_stride0, out_val, mask=mask)


def linspace(
    start,
    end,
    steps,
    *,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False
) -> torch.Tensor:
    logging.debug("GEMS LINSPACE")
    assert steps >= 1, "steps must be >= 1"

    if out is None:
        out = torch.empty(
            steps,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
    if steps == 1:
        return torch.fill(out, start)
    else:
        mid = steps // 2
        step_size = (float(end) - float(start)) / (steps - 1)
        BLOCK_SIZE = 128
        grid = (triton.cdiv(steps, BLOCK_SIZE),)
        linspace_kernel[grid](
            out, out.stride(0), start, mid, end, step_size, steps, BLOCK_SIZE=BLOCK_SIZE
        )
        return out
