import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_CO": 16}, num_stages=5),
    ],
    key=[
        "n_out",
        "c_out",
    ],
)
@triton.heuristics(
    values={
        "BLOCK_N": lambda args: args["c_input"],
        "BLOCK_W": lambda args: args["width_kernel"],
        "BLOCK_H": lambda args: args["height_kernel"],
        "BLOCK_OUT_WEIGHT": lambda args: args["height_kernel"]
        * args["c_input"]
        * args["width_kernel"],
    },
)
@triton.jit
def conv2d_img2col(
    weight,
    input,
    out,
    width_input,
    height_input,
    height_kernel,
    width_kernel,
    c_input,
    n_out,
    stride_height,
    stride_width,
    padding_height,
    padding_width,
    height_out,
    width_out,
    groups,
    c_out,
    BLOCK_CO: tl.constexpr,
    BLOCK_OUT_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_cn = tl.program_id(0)
    pid_out = tl.program_id(2)
    pid_group = tl.program_id(1)

    offset_cn = pid_cn * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_cout = pid_out * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offset_ci = tl.arange(0, BLOCK_N)  # todo make power

    # n g ci h w
    input_cn_offset = (
        offset_cn[:, None, None, None, None]
        * c_input
        * groups
        * width_input
        * height_input
    )
    input_ci_offset = offset_ci[:, None, None] * height_input * width_input
    input_group_offset = (
        pid_group[None, None, None, None] * c_input * width_input * height_input
    )

    # weight shape ci kh kw g cout
    weight_ci_offset = (
        offset_ci[:, None, None, None, None]
        * height_kernel
        * width_kernel
        * groups
        * c_out
    )
    weight_kh_kw_offset = (
        tl.arange(0, BLOCK_H)[:, None, None, None] * width_kernel * groups * c_out
        + tl.arange(0, BLOCK_W)[:, None, None] * groups * c_out
    )
    weight_group_offset = pid_group[None, None] * c_out
    weight_cout_offset = offset_cout
    weight_offset = (weight_ci_offset + weight_kh_kw_offset) + (
        weight_group_offset + weight_cout_offset
    )

    mask_weight = (offset_ci[:, None, None, None, None] < c_input) and (
        (weight_group_offset + weight_cout_offset) < c_out * groups
    )

    weight_value = tl.load(weight + weight_offset, mask_weight, other=0)
    # reshape to dot
    weight_value = tl.reshape(weight_value, (BLOCK_OUT_WEIGHT, BLOCK_CO))

    accumulator = tl.zeros((BLOCK_M, BLOCK_CO), dtype=tl.float32)

    for h in range(0, height_out):
        for w in range(0, width_out):
            offset_h = h * stride_height + tl.arange(0, BLOCK_H)
            offset_w = w * stride_width + tl.arange(0, BLOCK_W)
            offset_mid = input_ci_offset + offset_h[:, None] * (width_input) + offset_w

            mask = offset_h[:, None] < height_input and offset_w < width_input
            mask = mask and offset_cn[:, None, None, None, None] < n_out
            input_offset = input_cn_offset + input_group_offset + offset_mid

            mid_value = tl.load(input + input_offset, mask, other=0)

            mid_value = tl.reshape(mid_value, (BLOCK_M, BLOCK_OUT_WEIGHT))  # groups =1

            accumulator = tl.dot(mid_value, weight_value, allow_tf32=False)

            out_offset = (
                offset_cn * groups * height_out * width_out
                + pid_group * height_out * width_out
                + h * width_out
                + w
            )[:, None] * c_out + offset_cout
            mask_out = offset_cn[:, None] < n_out and offset_cout < c_out
            tl.store(out + out_offset, accumulator, mask_out)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_HWO": 16, "BLOCK_CO": 16}, num_stages=5),
    ],
    key=[
        "n_out",
        "height_out",
        "width_out",
    ],
)
@triton.heuristics(
    values={
        "BLOCK_N": lambda args: args["c_input"],
        "BLOCK_W": lambda args: args["width_kernel"],
        "BLOCK_H": lambda args: args["height_kernel"],
        "BLOCK_OUT_WEIGHT": lambda args: args["height_kernel"]
        * args["c_input"]
        * args["width_kernel"],
        "BLOCK_CI": lambda args: args["c_input"],
    },
)
@triton.jit  # todo add pad with col2img together  c_input=c_kernel diffrent in c
def conv2d_col2img(
    input,
    out_grad,
    weight,
    width_input,
    height_input,
    height_kernel,
    width_kernel,
    c_input,
    n_out,
    stride_height,
    stride_width,
    height_out,
    width_out,
    groups,
    c_out,
    BLOCK_CI: tl.constexpr,
    BLOCK_OUT_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_HWO: tl.constexpr,
):
    # init pid and offset
    pid_cn = tl.program_id(0)
    offset_cn = pid_cn * BLOCK_M + tl.arange(0, BLOCK_M)

    pid_groups = tl.program_id(1)

    offset_cout = tl.arange(0, BLOCK_CO)

    offset_ci = tl.arange(0, BLOCK_N)

    input_ci_offset = offset_ci[:, None, None] * height_input * width_input

    # first dot for all

    # load out_grad n groups  ho wo  cout
    out_grad_cn_offset = (
        offset_cn[:, None, None, None] * groups * height_out * width_out * c_out
    )
    out_grad_groups_offset = (
        pid_groups[None, None, None] * height_out * width_out * c_out
    )
    # out_grad_hw_offset = offset_hwo[:,None] * c_out
    out_grad_cout_offset = offset_cout
    out_grad_offset1 = (
        out_grad_cn_offset + out_grad_groups_offset + out_grad_cout_offset
    )

    out_grad_mask1 = offset_cn[:, None, None, None] < n_out
    # out_grad_mask = out_grad_mask and out_grad_cout_offset < c_out

    # load weight groups c_out ci h w
    weight_goups_offset = (
        pid_groups[None, None, None, None, None]
        * c_out
        * c_input
        * height_kernel
        * width_kernel
    )
    weight_cout_offset = (
        offset_cout[:, None, None, None] * c_input * height_kernel * width_kernel
    )
    weight_cin_offset = offset_ci[:, None, None] * height_kernel * width_kernel
    weight_kh_kw_offset = tl.arange(0, BLOCK_H)[:, None] * width_kernel + tl.arange(
        0, BLOCK_W
    )
    weight_offset1 = (
        weight_goups_offset
        + weight_cout_offset
        + weight_cin_offset
        + weight_kh_kw_offset
    )
    weight_mask1 = (
        offset_ci[:, None, None] < c_input
        and pid_groups[None, None, None, None, None] < groups
    )

    # BLOCK_M Groups, BLOCK_OUT_WEIGHT =ci h w
    input_cn_offset = (
        offset_cn[:, None, None, None, None]
        * groups
        * c_input
        * height_input
        * width_input
    )

    input_groups_offset = (
        pid_groups[None, None, None, None] * c_input * height_input * width_input
    )
    # second col2img
    # BLOCK_HWO
    # n groups ho wo ci h w
    # accumulator = tl.zeros((BLOCK_M, BLOCK_OUT_WEIGHT), dtype=tl.float32)
    for h in range(0, height_out):
        for w in range(0, width_out):
            # dot to get input
            out_grad_hw_offset = (h * width_out + w)[:, None] * c_out
            out_grad_offset = out_grad_hw_offset + out_grad_offset1

            accumulator = tl.zeros((BLOCK_M, BLOCK_OUT_WEIGHT), dtype=tl.float32)
            weight_offset = weight_offset1
            for k in range(0, tl.cdiv(c_out, BLOCK_CO)):
                # n_out ho wo c_out
                out_grad_mask = out_grad_mask1 and (
                    (out_grad_cout_offset) < (c_out - k * BLOCK_CO)
                )
                out_grad_value = tl.load(
                    out_grad + out_grad_offset, out_grad_mask, other=0
                )
                out_grad_value = tl.reshape(out_grad_value, (BLOCK_M, BLOCK_CO))
                # groups cout ci h w
                weight_mask = weight_mask1 and (
                    (offset_cout[:, None, None, None]) < (c_out - k * BLOCK_CO)
                )
                weight_value = tl.load(weight + weight_offset, weight_mask, other=0)
                weight_value = tl.reshape(weight_value, (BLOCK_CO, BLOCK_OUT_WEIGHT))

                accumulator += tl.dot(out_grad_value, weight_value, allow_tf32=False)

                out_grad_offset += BLOCK_CO
                weight_offset += BLOCK_CO * c_input * height_kernel * width_kernel

            col_value = tl.reshape(
                accumulator, (BLOCK_M, 1, BLOCK_CI, BLOCK_H, BLOCK_W)
            )

            offset_h = h * stride_height + tl.arange(0, BLOCK_H)  # - padding_height
            offset_w = w * stride_width + tl.arange(0, BLOCK_W)  # - padding_width
            offset_mid = input_ci_offset + offset_h[:, None] * (width_input) + offset_w

            mask = offset_h[:, None] < height_input and offset_w < width_input
            # mask = mask and h * width_out + w < BLOCK_HWO
            mask = mask and offset_cn[:, None, None, None, None] < n_out
            mask = mask and offset_ci[:, None, None] < c_input
            input_offset = input_cn_offset + input_groups_offset + offset_mid

            # col_value = tl.reshape(accumulator, (BLOCK_M, 1, BLOCK_CI, BLOCK_H, BLOCK_W))
            input_value = tl.load(input + input_offset, mask, other=0)
            mid = col_value + input_value

            tl.store(input + input_offset, mid, mask)


class Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        logging.debug("GEMS CONV2D")
        assert input.ndim == 3 or input.ndim == 4, "Invalid input tensor."
        assert weight.ndim == 3 or weight.ndim == 4, "Invalid weight tensor."
        if input.ndim == 4:
            n_out, c_input, height_input, width_input = list(input.shape)
        else:
            c_input, height_input, width_input = list(input.shape)
            n_out = 1
        ctx.save_for_backward(weight)
        if weight.ndim == 3:
            c_kernel, height_kernel, width_kernel = list(weight.shape)
            c_out = 1
        else:
            c_out, c_kernel, height_kernel, width_kernel = list(weight.shape)
            weight = weight.permute(1, 2, 3, 0).contiguous()

        assert c_kernel == c_input / groups, "Invalid channel value in input and weight"
        # todo make sure tuple is 2d
        if isinstance(stride, list):
            stride_height, stride_width = stride
        else:
            stride_height = stride_width = stride

        if isinstance(padding, list):
            padding_height, padding_width = padding
        else:
            padding_height = padding_width = padding

        height_out = (
            height_input + 2 * padding_height - height_kernel
        ) // stride_height + 1
        width_out = (width_input + 2 * padding_width - width_kernel) // stride_width + 1

        padded_input = torch.nn.functional.pad(
            input,
            (padding_height, padding_height, padding_width, padding_width),
            "constant",
            0,
        )
        padded_input = padded_input.contiguous()

        _, _, padded_height_input, padded_width_input = list(padded_input.shape)

        out = torch.zeros(
            n_out,
            groups,
            height_out,
            width_out,
            int(c_out / groups),
            dtype=input.dtype,
            device=input.device,
        )
        groups = groups

        grid = lambda meta: (
            triton.cdiv(n_out, meta["BLOCK_M"]),
            groups,
            triton.cdiv(int(c_out / groups), meta["BLOCK_CO"]),
        )

        conv2d_img2col[grid](
            weight,
            padded_input,
            out,
            padded_width_input,
            padded_height_input,
            height_kernel,
            width_kernel,
            c_kernel,
            n_out,
            stride_height,
            stride_width,
            padding_height,
            padding_width,
            height_out,
            width_out,
            groups,
            int(c_out / groups),
        )

        out = out.permute(0, 1, 4, 2, 3).contiguous()
        out = torch.flatten(out, start_dim=1, end_dim=2)

        # weight_reshape = weight.reshape(-1, c_out)
        # ctx.save_for_backward(weight_reshape)
        ctx.n_out = n_out
        ctx.c_kernel = c_kernel
        ctx.height_input = height_input
        ctx.width_input = width_input
        ctx.c_out = int(c_out / groups)
        ctx.height_kernel = height_kernel
        ctx.width_kernel = width_kernel
        ctx.padded_height_input = padded_height_input
        ctx.padded_width_input = padded_width_input
        ctx.height_out = height_out
        ctx.width_out = width_out
        ctx.stride_height = stride_height
        ctx.stride_width = stride_width

        ctx.padding_height = padding_height
        ctx.padding_width = padding_width
        ctx.dtype = input.dtype
        ctx.device = input.device
        ctx.groups = groups
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS CONV2D VJP")
        (weight,) = ctx.saved_tensors
        c_kernel = ctx.c_kernel
        groups = ctx.groups
        padded_height_input = ctx.padded_height_input
        padded_width_input = ctx.padded_width_input
        n_out = ctx.n_out
        dtype = ctx.dtype
        device = ctx.device
        groups = ctx.groups

        padded_input = torch.zeros(
            n_out,
            c_kernel * groups,
            padded_height_input,
            padded_width_input,
            dtype=dtype,
            device=device,
        )
        c_out = ctx.c_out  # (origin cout divide groups)
        grid = lambda meta: (
            triton.cdiv(n_out, meta["BLOCK_M"]),
            groups,
        )
        height_kernel = ctx.height_kernel
        width_kernel = ctx.width_kernel
        stride_height = ctx.stride_height
        stride_width = ctx.stride_width
        height_out = ctx.height_out
        width_out = ctx.width_out

        stride_height = ctx.stride_height

        # out_grad n groups cout ho wo

        out_grad = out_grad.reshape(
            n_out, groups, c_out, height_out, width_out
        ).contiguous()
        out_grad = out_grad.permute(0, 1, 3, 4, 2).contiguous()

        # weight groups co ci h w

        # input_col = mm(out_grad, weight_reshape.T)

        # return dx,None,None,None,None,None,None
        conv2d_col2img[grid](
            padded_input,
            out_grad,
            weight,
            padded_width_input,
            padded_height_input,
            height_kernel,
            width_kernel,
            c_kernel,
            n_out,
            stride_height,
            stride_width,
            height_out,
            width_out,
            groups,
            c_out,
        )
        height_input = ctx.height_input
        width_input = ctx.width_input
        padding_height = ctx.padding_height
        padding_width = ctx.padding_width
        return (
            padded_input[
                :,
                :,
                padding_height : height_input + padding_height,
                padding_height : width_input + padding_width,
            ],
            None,
            None,
            None,
            None,
            None,
            None,
        )


# todo test SymInt[2] of stride or padding
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return Conv2d.apply(input, weight, bias, stride, padding, dilation, groups)
