from typing import Tuple

import torch

Shape = Tuple[int]


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


def is_expandable_to(s1: Shape, s2: Shape) -> Shape:
    s1_ndim = len(s1)
    s2_ndim = len(s2)
    if s1_ndim > s2_ndim:
        return False
    for i in range(s1_ndim):
        size = s1[s1_ndim - i - 1]
        target_size = s2[s2_ndim - i - 1]
        if size != target_size and size != 1:
            return False
    return True


"""
def infer_size(shape, numel):
    new_size = 1
    infer_dim = -1
    for dim in range(0, len(shape)):
        if (shape[dim] == -1):
            infer_dim = dim
        elif (shape[dim] >= 0):
            new_size *= shape[dim]
        else:
            raise "Invalid shape dimension"
    if infer_dim and new_size > 0 and (numel % new_size == 0):
        shape[infer_dim] = numel // new_size
    return shape


def expand_tensor_list(to_expand):
    first = True
    sizes = []
    for i in range(0, len(to_expand)):
        if first:
            sizes = list(to_expand[i].shape)
            first = False
        else:
            sizes = infer_size(sizes, len(to_expand[i].shape))
    print("sizes", sizes)
"""


def infer_size(indices, shape):
    indexing_shape = list(indices.shape)
    if len(indexing_shape) != len(shape):
        for i in range(len(indexing_shape), len(shape)):
            indexing_shape.append(shape[i])

    size = 1
    first = True
    for i in range(0, len(indices)):
        if isinstance(indices[i], tuple) or (
            torch.is_tensor(indices[i]) and indices[i].ndim > 1
        ):
            continue
        if first:
            size = indices[i].numel()
            first = False
        assert indices[i].numel() == size, "indices has invalid shape"
    return size


def pre_process(indices):
    indices_ = []
    for i in range(0, len(indices)):
        if indices[i].ndim > 1:
            if indices[i].size(0) != 1:
                indices_.append(tuple(torch.ravel(indices[i]).tolist()))
            else:
                indices_.append(torch.ravel(indices[i]))
        else:
            indices_.append(indices[i])
    return indices_


def build_single_mask(indices, shape, x_num):
    res = torch.full(shape, False)
    for i in range(0, x_num):
        unravel_offsets = tuple([(t[i] if torch.is_tensor(t) else t) for t in indices])
        res[unravel_offsets] = True
    return res


def index_put_(inp, indices, values, accumulate=False):
    # 先假设 indices 和 values 的数量都是 match 的，写一版
    inp_shape = list(inp.shape)
    values_shape = list(values.shape)
    print(is_expandable_to(values_shape, inp_shape))
    """
    assert (
        (torch.is_tensor(values) and values.ndim == 1)
    ), "masked_fill_ only supports a 1-dimensional values tensor"
    inp_shape = list(inp.shape)
    indexing_shape = []
    for i in range(0, len(indices)):
        indexing_shape.extend(list(indices[i].shape))
    if len(indices) != len(inp_shape):
        for i in range(len(indices), len(inp_shape)):
            indexing_shape.append(inp_shape[i])
    values_shape = broadcast(list(values.shape), indexing_shape)
    values_ = torch.broadcast_to(values, values_shape)
    # FIXME: Should we just judge if values can broadcast to indices_shape or not?
    print(values_)
    """
    """
    x_num = infer_size(indices, inp_shape)
    indices_ = pre_process(indices)
    if (values.numel() == 1):
        val = values.item()
        mask = build_single_mask(indices_, inp_shape, x_num)
        return torch.masked_fill(inp, mask, val)
    else:
        # 对每个 value 都执行一遍 masked_fill
        for i in range(0, x_num):
            mask = torch.full(inp_shape, False)
            val = values[i].item()
            unravel_offsets = tuple([(t[i] if torch.is_tensor(t) else t) for t in indices_])
            mask[unravel_offsets] = True
            inp = torch.masked_fill(inp, mask, val)
    """
    return inp


inp1 = torch.arange(1, 19).reshape(3, 3, 2)
inp2 = torch.arange(1, 19).reshape(3, 3, 2)
indices = [
    torch.tensor([[1], [2]]),
    torch.tensor([0, 2]),
]
values = torch.tensor([100, 400])
torch_res = torch.index_put_(inp2, indices, values)
# print("torch_res", torch_res)
triton_res = index_put_(inp1, indices, values)
# print("triton_res", triton_res)
# print(torch.allclose(triton_res,torch_res))
