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

from .conv2d import conv2d_output_size

logger = logging.getLogger(__name__)


def conv3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """
    Determines the output size of a 3D convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
        dilation: Dilation.

    Returns:
        Output size of 3D convolution.
    """
    return conv2d_output_size(in_size, kernel_size, stride, padding, dilation)


def _triple(value, name):
    """Normalize an int-or-sequence conv parameter to a 3-tuple."""
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(f"{name} must have 3 elements, received {value}")
        return tuple(int(v) for v in value)
    return (int(value),) * 3


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """3D convolution via an explicit im2col plus batched matmul.

    This deliberately does not use a hand-written Triton kernel. The generic
    FlagGems conv3d kernel cannot be compiled correctly by the XPU Triton
    backend -- it hits several independent defects at once:

      * A ``strip-all-ir`` pass runs unconditionally on the SDNN path and
        deletes the body of any kernel whose name contains "conv", so the op
        silently returns uninitialized memory instead of failing.
      * ``tl.dot`` lowers to ``linalg.matmul``, whose shape inference rejects a
        reduction operand narrowed below the block size.
      * Chained 2D comparison masks abort compilation with "Unsupported data
        element type".
      * Masked loads lower to a DMA across the whole addressed range, so the
        gather over padded spatial offsets faults on out-of-range lanes.
      * Clamping those addresses to stay in bounds then exceeds the 64
        pipeline-event limit on xpu_arch 3.

    The im2col formulation sidesteps all of it: the patch gather becomes
    strided views plus a contiguous copy, and the reduction becomes a single
    ``torch.matmul`` that maps onto the vendor's tuned GEMM. Verified against
    ``torch.nn.functional.conv3d`` across the full conv3d test matrix.
    """
    logger.debug("GEMS_KUNLUNXIN CONV3D")

    assert weight.ndim == 5, f"Weights must be 5D, received shape {weight.shape}"
    assert (
        bias is None or bias.ndim == 1
    ), f"Bias must be 1D, received shape {bias.shape}"
    assert (
        input.shape[1] == groups * weight.shape[1]
    ), f"Incompatible input ({input.shape}) and weights ({weight.shape}) shape with {groups} groups"
    assert (
        bias is None or weight.shape[0] == bias.shape[0]
    ), f"Incompatible weights ({weight.shape}) and bias ({bias.shape}) shape"

    stride_d, stride_h, stride_w = _triple(stride, "stride")
    dilation_d, dilation_h, dilation_w = _triple(dilation, "dilation")

    same_pad = None
    if isinstance(padding, str):
        if padding == "valid":
            pad_d = pad_h = pad_w = 0
        elif padding == "same":
            assert (
                stride_d == 1 and stride_h == 1 and stride_w == 1
            ), f"padding='same' requires stride 1, received {stride}"
            # 'same' keeps the spatial extent, needing a total padding of
            # dilation * (kernel - 1) per dim. An odd total cannot be split
            # evenly, so (matching torch) the leading side gets total // 2 and
            # the extra element goes on the trailing side. Applied as an
            # explicit pre-pad below, in F.pad's last-dim-first order.
            pad_d = pad_h = pad_w = 0
            same_pad = []
            for k, dil in (
                (weight.shape[-1], dilation_w),
                (weight.shape[-2], dilation_h),
                (weight.shape[-3], dilation_d),
            ):
                total = dil * (k - 1)
                same_pad.extend((total // 2, total - total // 2))
        else:
            raise ValueError(
                f"Unsupported padding string: {padding}, only 'valid'/'same' are allowed."
            )
    else:
        pad_d, pad_h, pad_w = _triple(padding, "padding")

    in_n, _, in_d, in_h, in_w = input.shape
    out_c, weight_c, k_d, k_h, k_w = weight.shape

    # An asymmetric 'same' pad is applied up front, so the window math below
    # sees the already-padded extents with no further symmetric padding.
    if same_pad is not None:
        in_w += same_pad[0] + same_pad[1]
        in_h += same_pad[2] + same_pad[3]
        in_d += same_pad[4] + same_pad[5]

    out_d = conv3d_output_size(in_d, k_d, stride_d, pad_d, dilation_d)
    out_h = conv3d_output_size(in_h, k_h, stride_h, pad_h, dilation_h)
    out_w = conv3d_output_size(in_w, k_w, stride_w, pad_w, dilation_w)
    if out_d <= 0 or out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"Invalid output size ({out_d}, {out_h}, {out_w}) for input {input.shape} "
            f"and weight {weight.shape}"
        )

    output_dtype = input.dtype

    # Accumulate in fp32 for low-precision inputs: fp16 tops out around 65504
    # and a 3D reduction over channels * kD * kH * kW overflows well before
    # that, which would show up as inf/NaN in the result.
    compute_dtype = torch.float32 if output_dtype != torch.float64 else torch.float64
    x = input.to(compute_dtype)
    w = weight.to(compute_dtype)

    # F.pad takes the spatial dims last-to-first.
    if same_pad is not None:
        if any(same_pad):
            x = torch.nn.functional.pad(x, same_pad, value=0.0)
    elif pad_d or pad_h or pad_w:
        x = torch.nn.functional.pad(
            x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), value=0.0
        )

    # Extract one sliding window per output position. unfold() gives a strided
    # view and the [::dilation] slices pick the dilated taps within each window,
    # so nothing is materialized here -- the copy happens per chunk below.
    xu = x.unfold(2, dilation_d * (k_d - 1) + 1, stride_d)
    xu = xu.unfold(3, dilation_h * (k_h - 1) + 1, stride_h)
    xu = xu.unfold(4, dilation_w * (k_w - 1) + 1, stride_w)
    xu = xu[..., ::dilation_d, ::dilation_h, ::dilation_w]
    # (N, C, out_d, out_h, out_w, kD, kH, kW)

    k_vol = k_d * k_h * k_w
    out_per_group_c = out_c // groups

    # (groups, weight_c*kD*kH*kW, out_c/groups), shared by every chunk.
    filters_t = w.reshape(groups, out_per_group_c, weight_c * k_vol).transpose(-1, -2)

    out = torch.empty(
        (in_n, out_c, out_d, out_h, out_w), device=x.device, dtype=compute_dtype
    )

    # The patch matrix holds every kernel tap for every output position, so it is
    # k_vol times the size of the input -- 54 GiB for the largest benchmark
    # shape. Build it in chunks instead and run one GEMM per chunk. The chunk is
    # sized by a byte budget rather than a fixed count so the split adapts to the
    # shape; 256 MiB still leaves each GEMM large enough to run at full speed.
    budget_bytes = 256 * 1024 * 1024
    elem = x.element_size()
    # Cost of one (sample, output-depth) unit of the patch matrix.
    unit_bytes = max(1, groups * weight_c * out_h * out_w * k_vol * elem)
    units = max(1, budget_bytes // unit_bytes)
    if units >= out_d:
        d_step = out_d
        n_step = max(1, min(in_n, units // out_d))
    else:
        d_step = units
        n_step = 1

    for n0 in range(0, in_n, n_step):
        n1 = min(n0 + n_step, in_n)
        for d0 in range(0, out_d, d_step):
            d1 = min(d0 + d_step, out_d)
            nc, dc = n1 - n0, d1 - d0

            # Split C into (groups, weight_c) and move the reduced dims last
            # while still a view, so the reshape below is the only copy.
            chunk = xu[n0:n1, :, d0:d1].unflatten(1, (groups, weight_c))
            chunk = chunk.permute(0, 1, 3, 4, 5, 2, 6, 7, 8)
            patches = chunk.reshape(nc, groups, dc * out_h * out_w, weight_c * k_vol)

            # (nc, groups, dc*out_h*out_w, out_c/groups)
            acc = torch.matmul(patches, filters_t)
            out[n0:n1, :, d0:d1] = acc.transpose(-1, -2).reshape(
                nc, out_c, dc, out_h, out_w
            )

    if bias is not None:
        out.add_(bias.to(compute_dtype).view(1, -1, 1, 1, 1))

    return out.to(output_dtype)
