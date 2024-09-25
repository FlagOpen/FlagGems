import logging

import torch

from ..ops import gather


def index_select(inp, dim, index):
    logging.debug("GEMS INDEX SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim

    # Handle the case where inp is a scalar
    if inp.ndim == 0:
        return torch.empty_like(inp).index_copy(0, index, inp.expand_as(index))

    # Reshape index to match the shape of the input tensor for gathering
    # Create a shape of [1, 1, ..., index.size(0), 1, ..., 1]
    # where the size at 'dim' is index.size(0)
    new_index_shape = [1] * inp.ndim
    new_index_shape[dim] = index.size(0)
    index_expanded = index.view(new_index_shape)

    # Expand index to match the full input tensor size
    # except for the selected dimension
    expanded_shape = list(inp.shape)
    expanded_shape[dim] = index.size(0)
    index_expanded = index_expanded.expand(expanded_shape)

    return gather(inp, dim, index_expanded)
