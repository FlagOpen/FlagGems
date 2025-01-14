import torch
import triton
import triton.language as tl

from ..runtime import torch_device_fn
from ..utils import libentry


@libentry()
@triton.jit()
def vdot_onestage_kernel(
    input,
    other,
    out,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, N, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = cols < N
        input_val = tl.load(input + cols, mask=mask).to(tl.float32)
        other_val = tl.load(other + cols, mask=mask).to(tl.float32)
        acc += input_val * other_val

    final_sum = tl.sum(acc, axis=0)
    tl.store(out, final_sum)


def vdot(input, other):
    N = input.numel()
    BLOCK_SIZE = 4096
    out = torch.empty([], device=input.device, dtype=input.dtype)
    with torch_device_fn.device(input.device):
        vdot_onestage_kernel[(1,)](input, other, out, N, BLOCK_SIZE)
    return out
