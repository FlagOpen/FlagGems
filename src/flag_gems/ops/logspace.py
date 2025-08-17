import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry
from ..utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

@libentry()
@triton.jit
def logspace_kernel(
    out_ptr,
    out_stride0,
    start,
    base,
    step_size,
    steps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < steps

    exponent = start + idx * step_size
    vals = tl.exp2(tl.log2(base.to(tl.float32)) * exponent.to(tl.float32))

    tl.store(out_ptr + idx * out_stride0, vals, mask=mask)


def logspace(
        start, end, steps, base=10.0, *,  dtype=None, layout=None, device=None, pin_memory=None
    ) -> torch.Tensor:
    logger.debug("GEMS LOGSPACE")
    assert steps >= 1, "steps must be >= 1"

    out = torch.empty(
        steps,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    if steps == 1:
        out = torch.fill(out, base**start)
    else:
        if isinstance(start, torch.Tensor):
            start = start.item()
        if isinstance(end, torch.Tensor):
            end = end.item()
        step_size = (float(end) - float(start)) / (steps - 1)
        BLOCK_SIZE = 128
        grid = (triton.cdiv(steps, BLOCK_SIZE),)
        logspace_kernel[grid](
            out, out.stride(0), start, base, step_size, steps, BLOCK_SIZE = BLOCK_SIZE
        )

    return out
