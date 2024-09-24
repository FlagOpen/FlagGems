import logging

import torch
import triton
from triton import language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic
from ..utils.shape_utils import c_contiguous_stride
from ..utils.tensor_wrapper import StridedBuffer


@pointwise_dynamic(num_inputs=1, promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def repeat_interleave_self_int(inp, repeats, dim=None, *, output_size=None):
    logging.debug("GEMS REPEAT_INTERLEAVE_SELF_INT")
    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if (dim < -inp.ndim) or (dim >= inp.ndim):
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )
    inp_shape = list(inp.shape)
    inp_stride = list(inp.stride())
    output_shape = list(inp.shape)

    if dim < 0:
        dim = dim + len(inp_shape)

    output_shape[dim] *= repeats

    if output_size is not None and output_size != output_shape[dim]:
        raise RuntimeError(
            "repeat_interleave: Invalid output_size, expected {} but got {}".format(
                output_shape[dim], output_size
            )
        )

    output = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    if repeats == 0:
        return output

    in_view_stride = inp_stride[: dim + 1] + [0] + inp_stride[dim + 1 :]
    out_view_shape = inp_shape[: dim + 1] + [repeats] + inp_shape[dim + 1 :]
    out_view_stride = c_contiguous_stride(out_view_shape)

    in_view = StridedBuffer(inp, out_view_shape, in_view_stride)
    out_view = StridedBuffer(output, out_view_shape, out_view_stride)
    ndim = len(out_view_shape)
    copy_func.instantiate(ndim)(in_view, out0=out_view)
    return output


@triton.jit
def repeat_interleave_tensor_kernel(inputs_ptr, repeats_ptr, cumsum_ptr, out_ptr, size):
    pid = tl.program_id(0)
    mask = pid < size
    cumsum = tl.load(cumsum_ptr + pid, mask, other=0)
    inputs = tl.load(inputs_ptr + pid + 1, mask)
    repeats = tl.load(repeats_ptr + pid + 1, mask)

    tl.device_assert(repeats >= 0, "repeats can not be negative")

    out_offsets = cumsum
    for _ in range(repeats):
        tl.store(out_ptr + out_offsets, inputs)
        out_offsets += 1


def repeat_interleave_tensor(repeats, *, output_size=None):
    logging.debug("GEMS REPEAT_INTERLEAVE_TENSOR")

    assert repeats.ndim == 1, "repeat_interleave only accept 1D vector as repeat"

    cumsum = repeats.cumsum(axis=0)
    result_size = cumsum[-1].item()

    assert cumsum[0].item() >= 0, "repeats can not be negative"
    assert result_size >= 0, "repeats can not be negative"

    out = torch.zeros((result_size,), dtype=repeats.dtype, device=repeats.device)
    size = repeats.size(0)
    inputs = torch.arange(0, size, dtype=repeats.dtype, device=repeats.device)

    grid = (size - 1,)
    repeat_interleave_tensor_kernel[grid](
        inputs, repeats, cumsum, out, size, num_warps=1
    )
    return out
