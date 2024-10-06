import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 16, "BLOCK_O": 16, "BLOCK_C_IN": 16}, num_stages=5),
    ],
    key=[
        "n_out",
        "c_out",
    ],
)
@triton.heuristics(
    values={
        # "BLOCK_C_IN": lambda args: triton.next_power_of_2(args["c_input"]),
        "BLOCK_W": lambda args: args["width_kernel"],
        "BLOCK_OUT_WEIGHT": lambda args: args["BLOCK_C_IN"] * args["width_kernel"],
    },
)
@triton.jit
def conv1d_compute(
    padded_input,
    weight,
    out,
    padded_width_input,
    width_kernel,
    c_input,
    n_out,
    c_out,
    stride_width,
    width_out,
    BLOCK_O: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_OUT_WEIGHT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_cout = tl.program_id(1)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_cout = pid_cout * BLOCK_O + tl.arange(0, BLOCK_O)

    accumulator = tl.zeros((BLOCK_N, BLOCK_O), dtype=tl.float32)
    for i in range(0, width_out):
        input_offset = (
            (offset_n * c_input * padded_width_input)[:, None, None]
            + (tl.arange(0, BLOCK_C_IN) * padded_width_input)[:, None]
            + i * stride_width
            + tl.arange(0, BLOCK_W)
        )
        mask_input = (offset_n)[:, None, None] < n_out and (tl.arange(0, BLOCK_C_IN))[
            :, None
        ] < c_input
        weight_offset = (
            (tl.arange(0, BLOCK_C_IN) * width_kernel * c_out)[:, None, None]
            + (tl.arange(0, BLOCK_W) * c_out)[:, None]
            + offset_cout
        )
        mask_weight = (offset_cout) < c_out and (tl.arange(0, BLOCK_C_IN))[
            :, None, None
        ] < c_input

        # load
        input_value = tl.load(padded_input + input_offset, mask=mask_input, other=0)
        weight_value = tl.load(weight + weight_offset, mask=mask_weight, other=0)
        input_value = tl.reshape(input_value, (BLOCK_N, BLOCK_OUT_WEIGHT))
        weight_value = tl.reshape(weight_value, (BLOCK_OUT_WEIGHT, BLOCK_O))
        accumulator = tl.dot(input_value, weight_value, allow_tf32=False)
        out_offset = (
            (offset_n * c_out * width_out)[:, None] + (offset_cout) * width_out + i
        )
        mask_out = (offset_n)[:, None] < n_out and offset_cout < c_out
        tl.store(out + out_offset, accumulator, mask=mask_out)


class Conv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        logging.debug("GEMS CONV1D")
        assert input.ndim == 2 or input.ndim == 3, "Invalid input tensor."
        assert weight.ndim == 2 or weight.ndim == 3, "Invalid weight tensor."
        if input.ndim == 3:
            n_out, c_input, width_input = list(input.shape)
        else:
            c_input, width_input = list(input.shape)
            n_out = 1

        if weight.ndim == 3:
            c_out, c_kernel, width_kernel = list(weight.shape)
            weight = weight.permute(1, 2, 0).contiguous()
        else:
            c_kernel, width_kernel = list(weight.shape)
            c_out = 1
        assert c_kernel == c_input, "Invalid channel value in input and weight"
        # todo make sure tuple is 2d
        if isinstance(stride, list):
            stride_width = stride[0]
        else:
            stride_width = stride

        if isinstance(padding, list):
            padding_width = padding[0]
        else:
            padding_width = padding

        width_out = (width_input + 2 * padding_width - width_kernel) // stride_width + 1

        padded_input = torch.nn.functional.pad(
            input, (padding_width, padding_width), "constant", 0
        )
        padded_input = padded_input.contiguous()
        _, _, padded_width_input = list(padded_input.shape)
        out = torch.zeros(
            n_out, c_out, width_out, dtype=input.dtype, device=input.device
        )
        grid = lambda meta: (
            triton.cdiv(n_out, meta["BLOCK_N"]),
            triton.cdiv(c_out, meta["BLOCK_O"]),
        )
        conv1d_compute[grid](
            padded_input,
            weight,
            out,
            padded_width_input,
            width_kernel,
            c_input,
            n_out,
            c_out,
            stride_width,
            width_out,
        )

        return out


# todo test SymInt[2] of stride or padding
def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return Conv1d.apply(input, weight, bias, stride, padding, dilation, groups)
