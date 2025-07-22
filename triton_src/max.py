import logging
from collections import namedtuple
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def max_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)


@libentry()
@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tle.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    min_value = get_dtype_min(dtype)
    result_value = tl.full([BLOCK_M], value=min_value, dtype=acc_type)
    result_index = tl.zeros([BLOCK_M], dtype=tl.int64)
    for i in range(0, N, BLOCK_N):
        n_offset = i + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        # set mask
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        max_value, max_index = tl.max(inp_vals, axis=1, return_indices=True)
        update_mask = max_value > result_value
        result_value = tl.where(update_mask, max_value, result_value)
        result_index = tl.where(update_mask, i + max_index, result_index)
    mask1 = m_offset < M
    offset_index = m_offset
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)
