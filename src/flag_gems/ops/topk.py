import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import (
    _get_sort_dim,
    _indicator,
    _is_power_of_two,
    _log2,
    _take_slice,
    zeros_like,
)


@triton.jit
def topk_stage1_kernel(
    Y,  # pointer to the output
    INDEX,  # pointer to the output
    X,  # pointer to the input
    k,
    N: tl.constexpr,  # number of columns in X
    CHUNK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * k
    INDEX += pid * k

    offset = pid * CHUNK_SIZE
    X += offset

    cols = tl.arange(0, CHUNK_SIZE)
    mask = (offset + cols) < N

    x_val = tl.load(X + cols, mask=mask, other=-10000.0)
    for k_idx in range(k):
        chunk_max_val = tl.max(x_val)
        chunk_max_idx = tl.argmax(x_val, axis=0)
        tl.store(Y + k_idx, chunk_max_val)
        tl.store(INDEX + k_idx, chunk_max_idx + offset)
        x_val = tl.where(cols == chunk_max_idx, -10000, x_val)


@triton.jit
def _compare_and_swap(x, ids, desc_mask, n_dims: core.constexpr, idx: core.constexpr):
    l_slice = _take_slice(x, n_dims, idx, 0)
    r_slice = _take_slice(x, n_dims, idx, 1)

    x_int = x
    l_int = l_slice
    r_int = r_slice

    l_idx = _take_slice(ids, n_dims, idx, 0)
    r_idx = _take_slice(ids, n_dims, idx, 1)

    idx_int = ids
    l_int_idx = l_idx
    r_int_idx = r_idx

    # tl.device_print(x)
    if x.dtype.is_floating():
        if core.constexpr(x.dtype.primitive_bitwidth) == 16:
            dtype_int = core.int16
        elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
            dtype_int = core.int32
        elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
            dtype_int = core.int64
        else:
            raise ValueError("Unsupported dtype")
        x_int = x.to(dtype_int, bitcast=True)
        l_int = l_slice.to(dtype_int, bitcast=True)
        r_int = r_slice.to(dtype_int, bitcast=True)

    desc_mask = desc_mask.to(x_int.dtype)
    zero = zeros_like(x_int)
    cond = (l_slice > r_slice) ^ desc_mask
    y = x_int ^ core.where(cond, l_int ^ r_int, zero)
    y = y.to(x.dtype, bitcast=True)

    idx = idx_int ^ core.where(cond, l_int_idx ^ r_int_idx, zeros_like(ids))
    return y, idx


@triton.jit
def _bitonic_merge(
    x,
    ids,
    n_dims: core.constexpr,
    active_dims: core.constexpr,
    order_type: core.constexpr,
):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    core.static_assert(active_dims <= n_dims)

    if order_type == 2:
        desc_mask = _indicator(n_dims, active_dims, 1)
    else:
        desc_mask = order_type

    for i in core.static_range(active_dims):
        x, ids = _compare_and_swap(x, ids, desc_mask, n_dims, active_dims - 1 - i)

    return x, ids


@triton.jit
def argsort(x, ids, dim: core.constexpr = None, descending: core.constexpr = 0):
    core.static_assert(_is_power_of_two(x.shape[_get_sort_dim(dim, x.shape)]))
    core.static_assert(_is_power_of_two(x.numel))
    # reshape the tensor to have all dimensions be 2.
    # TODO: We shouldn't have to change the dimensions not sorted.
    y = core.reshape(x, [2] * _log2(x.numel))
    y_ids = core.reshape(ids, [2] * _log2(ids.numel))

    for i in core.static_range(1, _log2(x.shape[_get_sort_dim(dim, x.shape)]) + 1):
        y, y_ids = _bitonic_merge(
            y,
            y_ids,
            _log2(x.numel),
            i,
            (descending if (i == _log2(x.shape[_get_sort_dim(dim, x.shape)])) else 2),
        )

    x = core.reshape(y, x.shape)
    ids = core.reshape(y_ids, ids.shape)

    return x, ids


@triton.jit
def topk_stage2_kernel(
    Y,  # pointer to the output
    INDEX,  # pointer to the output
    CHUNK_X,  # pointer to the input
    CHUNK_INDEX,  # pointer to the input
    k: tl.constexpr,
    N: tl.constexpr,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    chunk_x_val = tl.load(CHUNK_X + cols, mask=mask, other=-10000.0)
    chunk_index_val = tl.load(CHUNK_INDEX + cols, mask=mask, other=-10000)

    # for k_idx in range(k):
    #     chunk_max_val = tl.max(chunk_x_val)
    #     chunk_max_idx = tl.argmax(chunk_x_val, axis=0)

    #     tl.store(Y + k_idx, chunk_max_val)
    #     chunk_index_val = tl.load(CHUNK_INDEX + chunk_max_idx)
    #     tl.store(INDEX + k_idx, chunk_index_val)

    #     chunk_x_val = tl.where(cols == chunk_max_idx, -10000, chunk_x_val)

    sorted_chunk_x, sorted_chunk_index = argsort(
        chunk_x_val, chunk_index_val, descending=True
    )
    tl.store(Y + cols, sorted_chunk_x, mask=cols < k)
    tl.store(INDEX + cols, sorted_chunk_index, mask=cols < k)
