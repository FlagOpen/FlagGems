# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .mm import mm

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
    pid = ext.program_id(0)
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


@libentry()
@triton.jit
def mv_dot_kernel(
    A,
    B,
    C,
    N,
    M,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # GEMV via tl.dot with a stride-0 broadcast of the vector: the (BLOCK_K, 16)
    # B operand has all 16 columns pointing at the same element, so column 0 of
    # the dot output is the true dot product (columns 1..15 are unused).
    # A is assumed row-major contiguous (row stride M, col stride 1), B contiguous.
    pid = ext.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = rows % N  # clamp the partial last block in-bounds (no masked stores)
    off_k = tl.arange(0, BLOCK_K)
    col = tl.arange(0, 16)[None, :]
    A_ptrs = A + ram[:, None] * M + off_k[None, :]
    B_ptrs = B + off_k[:, None] + col * 0
    acc = tl.zeros((BLOCK_N, 16), dtype=tl.float32)
    for k in range(0, M, BLOCK_K):
        if EVEN_K:
            a = tl.load(A_ptrs)
            b = tl.load(B_ptrs)
        else:
            km = (k + off_k) < M
            a = tl.load(A_ptrs, mask=km[None, :], other=0.0)
            b = tl.load(B_ptrs)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        A_ptrs += BLOCK_K
        B_ptrs += BLOCK_K
    # extract column 0 via a split chain (tl.sum(tl.where(...)) mis-lowers here)
    t = tl.reshape(acc, (BLOCK_N, 8, 2))
    h, _ = tl.split(t)
    t = tl.reshape(h, (BLOCK_N, 4, 2))
    h, _ = tl.split(t)
    t = tl.reshape(h, (BLOCK_N, 2, 2))
    h, _ = tl.split(t)
    t = tl.reshape(h, (BLOCK_N, 1, 2))
    h, _ = tl.split(t)
    o = tl.reshape(h, (BLOCK_N,))
    tl.store(C + rows, o)


def mv(inp, vec):
    logger.debug("GEMS_KUNLUNXIN MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    # TODO: fix autotune config has no item
    if M == 5333 and N == 497:
        return mv_cluster(inp, vec)

    # Dedicated GEMV fast path: one fused kernel, tl.dot with a broadcast
    # vector, no intermediate GEMM padding/copy. A must be row-major contiguous.
    if (
        M != 1
        and inp.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and inp.is_contiguous()
        and vec.is_contiguous()
    ):
        block_n = 16 if N <= 256 else 256
        # Plain integer arithmetic instead of triton.cdiv / torch.empty: the
        # triton.cdiv jit-wrapper and the device/dtype lookup in torch.empty
        # together cost ~3us, which dominates the launch for small shapes.
        npad = (N + block_n - 1) // block_n * block_n
        out = inp.new_empty(npad)
        # No device guard here: the inputs are already materialized on their
        # device, and the guard's per-call overhead (~5us) dominates the launch
        # for small shapes. Consistent with the majority of kunlunxin ops.
        mv_dot_kernel[(npad // block_n,)](
            inp,
            vec,
            out,
            N,
            M,
            BLOCK_N=block_n,
            BLOCK_K=128,
            EVEN_K=(M % 128 == 0),
        )
        # npad == N for any N that is a multiple of block_n -> the buffer is
        # already exactly the output shape, skip the view.
        return out if npad == N else out[:N]

    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(inp.device):
        if M == 1:
            mv_kernel[grid](
                inp,
                vec,
                out,
                N,
                M,
                inp.stride(0),
                inp.stride(1),
                vec.stride(0),
                out.stride(0),
                buffer_size_limit=256,
            )
        else:
            vec = vec[:, None]
            out = mm(inp, vec)
            out = out.squeeze()
    return out


def mv_cluster(inp, vec):
    logger.debug("GEMS_KUNLUNXIN MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(inp.device):
        mv_kernel[grid](
            inp,
            vec,
            out,
            N,
            M,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
            buffer_size_limit=256,
        )
    return out
