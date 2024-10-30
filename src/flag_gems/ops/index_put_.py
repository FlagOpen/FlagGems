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


def infer_size(a, b):
    dimsA = len(a)
    dimsB = len(b)
    ndim = max(dimsA, dimsB)
    expandedSize = []

    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        sizeA = a[dimA] if (dimA >= 0) else 1
        sizeB = b[dimB] if (dimB >= 0) else 1

        assert (
            (sizeA == sizeB) or (sizeA == 1) or (sizeB == 1)
        ), "shape mismatch: indexing tensors could not be broadcast together"
        expandedSize.append(sizeB if sizeA == 1 else sizeA)

    return expandedSize


def expand_outplace(to_expand):
    # expands a list of Tensors
    first = True
    sizes = []
    for i in range(0, len(to_expand)):
        if first:
            sizes = list(to_expand[i].shape)
            first = False
        else:
            sizes = infer_size(sizes, list(to_expand[i].shape))

    result = []
    for i in range(0, len(to_expand)):
        if list(to_expand[i].shape) == sizes:
            result.append(to_expand[i])
        else:
            result.append(broadcast(sizes, to_expand[i]))

    print(result)
    return result


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
    # Temporarily handle: The rank of tensors in indices are all 1-D
    inp_shape = list(inp.shape)
    values_shape = list(values.shape)
    assert is_expandable_to(
        values_shape, inp_shape
    ), "Value tensor's size should be broadcastable to input tensor. "
    assert (torch.is_tensor(values) and values.ndim == 1) or isinstance(
        values, int
    ), "index_put_ only support a 1-dimensional values tensor or a single value"
    indices = expand_outplace(indices)
    x_num = indices[0].size(0)

    indices_ = indices
    for i in range(len(indices_), inp.ndim, 1):
        cur_size = inp.size(i)
        meta = []
        for j in range(0, cur_size):
            meta.append([j] * cur_size)
        indices_.append(torch.tensor(meta))

    if values.numel() == 1:
        val = values.item()
        mask = build_single_mask(indices_, inp_shape, x_num)
        # Dispatch
        return torch.masked_fill(inp, mask, val)
    else:
        # 对每个 value 都执行一遍 masked_fill
        for i in range(0, x_num):
            mask = torch.full(inp_shape, False)
            val = values[i].item()
            unravel_offsets = tuple(
                [(t[i] if torch.is_tensor(t) else t) for t in indices_]
            )
            mask[unravel_offsets] = True
            inp = torch.masked_fill(inp, mask, val)
    return inp


inp1 = torch.arange(1, 19).reshape(3, 3, 2)
inp2 = torch.arange(1, 19).reshape(3, 3, 2)
indices = [torch.tensor([1, 2]), torch.tensor([0, 2]), torch.tensor([[0, 1], [0, 1]])]
values = torch.tensor([100, 400, 800])
torch_res = torch.index_put_(inp2, indices, values)
print("torch_res", torch_res)
triton_res = index_put_(inp1, indices, values)
print("triton_res", triton_res)
print(torch.allclose(triton_res, torch_res))
