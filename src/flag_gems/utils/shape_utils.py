import functools
import operator
from typing import Iterable, Tuple

import torch

Shape = Tuple[int]
Stride = Tuple[int]
MultiIndex = Tuple[int]
Perm = Tuple[int]


def broadcast(s1: Shape, s2: Shape) -> Shape:
    _s1, _s2 = s1, s2
    r1 = len(s1)
    if r1 == 0:
        return s2
    r2 = len(s2)
    if r2 == 0:
        return s1

    s1, s2 = (s1, s2) if r1 >= r2 else (s2, s1)
    r1, r2 = (r1, r2) if r1 >= r2 else (r2, r1)

    d = r1 - r2
    s = list(s1)

    for i in range(r2):
        if s1[d + i] == 1:
            s[d + i] = s2[i]
        elif s2[i] == 1:
            s[d + i] = s1[d + i]
        elif s2[i] == s1[d + i]:
            s[d + i] = s2[i]
        else:
            raise ValueError(f"Unbroadcastable {_s1} and {_s2}")
    s = tuple(s)
    return s


def broadcastable(s1: Shape, s2: Shape) -> bool:
    r1 = len(s1)
    if r1 == 0:
        return True
    r2 = len(s2)
    if r2 == 0:
        return True

    s1, s2 = (s1, s2) if r1 >= r2 else (s2, s1)
    r1, r2 = (r1, r2) if r1 >= r2 else (r2, r1)

    d = r1 - r2
    for i in range(r2):
        if s1[d + i] == 1 or s2[i] == 1 or s1[d + i] == s2[i]:
            continue
        return False
    return True


def broadcastable_to(s1: Shape, s2: Shape) -> bool:
    r1 = len(s1)
    if r1 == 0:
        return True
    r2 = len(s2)
    if r2 == 0:  # r1 > 0
        return False

    if r1 > r2:
        return False

    d = r2 - r1
    for i in range(r1):
        if s1[i] == 1 or s1[i] == s2[d + i]:
            continue
        return False
    return True


def broadcast_shapes(shapes: Iterable[Shape]) -> Shape:
    if len(shapes) == 0:
        return ()
    shape = shapes[0]
    for s in shapes[1:]:
        shape = broadcast(shape, s)
    return shape


def broadcasted_stride(shape: Shape, stride: Stride, new_shape: Shape) -> Stride:
    assert broadcastable_to(shape, new_shape)
    r1 = len(shape)
    r2 = len(new_shape)
    d = r2 - r1
    new_stride = [0 for _ in range(r2)]
    for i in range(r1):
        new_stride[d + i] = 0 if (shape[i] == 1 and new_shape[d + i] > 1) else stride[i]
    return tuple(new_stride)


def volume(shape: Shape) -> int:
    return functools.reduce(operator.mul, shape, 1)


def is_valid_perm(perm: Perm) -> bool:
    r = len(perm)
    sorted_axes = sorted(perm)
    for i in range(r):
        if sorted_axes[i] != i:
            return False
    return True


def unravel_index(linear_offset: int, shape: Shape) -> MultiIndex:
    multi_index = []
    r = len(shape)
    for i in range(r):
        s = shape[r - 1 - i]
        i = linear_offset % s
        linear_offset = linear_offset // s
        multi_index.append(i)
    return tuple(reversed(multi_index))


def c_contiguous_stride(shape: Shape) -> Stride:
    strides = []
    s = 1
    for size in reversed(shape):
        strides.append(s)
        s *= size

    return tuple(reversed(strides))


def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()


def size_in_bytes(a):
    return a.numel() * a.element_size()


def can_use_int32_index(a):
    INT32_MAX = torch.iinfo(torch.int32).max
    if a.is_contiguous():
        return size_in_bytes(a) <= INT32_MAX

    max_offset = 0
    for size, stride in zip(a.shape, a.stride()):
        max_offset += size * stride
        if max_offset > INT32_MAX:
            return False
    return True


def offsetCalculator(inp, idx, strides, dim, isInp):
    """
    Calculate the flat index(a.k.a offset) for a given ravel index in a multi-dimensional array.
    The formula can be seen in:
        - https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray
        - https://numpy.org/devdocs/user/basics.indexing.html#single-element-indexing


    Parameters:
        inp (tensor):               The input multi-dimensional array from which the offset is calculated.
        idx (tensor):               The linear index for which the offset is to be calculated.
        strides (list of int):      A list containing the stride lengths for each dimension of the input array.
        dim (int):                  The specific dimension for which the index offset needs to be calculated.
        isInp (bool):               A flag indicating whether the tensor 'inp' is the parameter 'self'
                                    in scatter/gather/index_* operators or not.

                                    In operators such as scatter/gather and index_*, when the input tensor 'inp'
                                    is the 'self' tensor to be processed, we may need to modify its offsets later.
                                    For instance, in the scatter operator, the offset is calculated using the formula:

                                        inp_offset = origin_offset - stride[dim] * n_dim + stride[dim] * index.

                                    In this case, we return the fixed part of the formula:

                                        origin_offset - stride[dim] * n_dim,

                                    to facilitate subsequent modifications.
                                    For other types of input 'inp', we return the complete calculation result
                                    of origin_offsets directly.


    Returns:
    The calculated offset. If isInp is True, the fixed offset is returned; otherwise, the origin offset is returned.


    Note:
    The function includes a comment suggesting the potential optimization of division and modulus operations,
    which may be beneficial if this function is called frequently.
    See also:
        - https://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html
        - Division by Invariant Integers Using Multiplication,
            Torbjörn Granlund and Peter L. Montgomery, 1994.
    """
    ndim = inp.ndim
    shape = list(inp.shape)
    offsets = 0
    idx_dim = 0
    for d in range(0, ndim):
        mod = idx % shape[d]
        add_on = mod * strides[d]
        offsets += add_on
        if d == dim:
            idx_dim = add_on
        idx = idx // shape[d]
    return (offsets) if not isInp else (offsets - idx_dim)


def restride_dim(src, dim, shape, step=0, storage_offset=None):
    strides = list(src.stride())
    strides[dim] *= step
    return src.as_strided(shape, strides, storage_offset)
