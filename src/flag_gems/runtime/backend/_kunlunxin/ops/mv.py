import logging

import triton
import triton.language as tl

from flag_gems.ops.mm import mm

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


def heur_block_n(args):
    return triton.next_power_of_2(triton.cdiv(args["N"], 12))


def heur_block_m(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["M"]), 4096)


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("mv"),
#     key=["M", "N"],
# )
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
        "BLOCK_M": heur_block_m,
    }
)
@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N: tl.constexpr,
    M: tl.constexpr,
    stride_an: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    buffer_size_limit: tl.constexpr,  # NOTE: `constexpr` so it can be used as a shape value.
):
    pid = tle.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_m = tl.arange(0, BLOCK_M)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        m_mask = m + offset_m < M
        a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm

    acc = tl.sum(acc, axis=1)
    C_ptrs = C + offset_n * stride_cn
    tl.store(C_ptrs, acc[:, None], mask=n_mask)


def mv(inp, vec):
    logger.debug("GEMS MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    with torch_device_fn.device(inp.device):
        vec = vec[:, None]
        out = mm(inp, vec)
    return out
