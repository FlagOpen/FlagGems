"""
Implementation of avg_pool2d using dynamic Triton code generation.
Supports all kernel sizes with an adaptive strategy for optimal performance.
"""

import importlib.util
import logging

import torch
import triton

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)

# Cache for dynamically generated kernels
_codegen_kernel_cache = {}


def generate_avg_pool2d_kernel_code(kernel_h, kernel_w, count_include_pad, code):
    """
    Generate Triton kernel source code with unrolled loops for specific kernel size.

    Adaptive strategy:
    - Small kernels (<=4x4): Full unroll with unique variables for better ILP
    - Large kernels (>4x4): Compact code with reused variables to reduce register pressure

    Args:
        kernel_h: Kernel height
        kernel_w: Kernel width
        count_include_pad: Whether to include padding in count
        code: IndentedBuffer to write code to

    Returns:
        IndentedBuffer with generated code
    """
    # Write imports
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()

    # Determine if this is a large kernel
    is_large_kernel = (kernel_h * kernel_w) > 16  # More than 4x4

    # Write kernel decorator and signature
    code.writeline("@triton.autotune(")
    code.writeline("    configs=[")
    code.writeline('        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),')
    code.writeline('        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),')
    code.writeline('        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),')
    code.writeline('        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),')
    code.writeline("    ],")
    code.writeline('    key=["N", "C", "H_out", "W_out"],')
    code.writeline(")")
    code.writeline("@triton.jit")
    kernel_name = f"avg_pool2d_kernel_{kernel_h}x{kernel_w}_pad{int(count_include_pad)}"
    code.writeline(f"def {kernel_name}(")
    code.writeline("    input_ptr,")
    code.writeline("    output_ptr,")
    code.writeline("    N,")
    code.writeline("    C,")
    code.writeline("    H,")
    code.writeline("    W,")
    code.writeline("    H_out,")
    code.writeline("    W_out,")
    code.writeline("    stride_h,")
    code.writeline("    stride_w,")
    code.writeline("    pad_h,")
    code.writeline("    pad_w,")
    code.writeline("    divisor_override,")
    code.writeline("    BLOCK_SIZE: tl.constexpr,")
    code.writeline("):")

    # Kernel body
    code.writeline("    # Get the program ID")
    code.writeline("    pid = tl.program_id(0)")
    code.newline()
    code.writeline("    # Calculate indices for this block")
    code.writeline("    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
    code.writeline("    mask = idx < N * C * H_out * W_out")
    code.newline()
    code.writeline("    # Convert flat index to (n, c, h_out, w_out)")
    code.writeline("    w_out = idx % W_out")
    code.writeline("    h_out = (idx // W_out) % H_out")
    code.writeline("    c = (idx // (W_out * H_out)) % C")
    code.writeline("    n = idx // (C * H_out * W_out)")
    code.newline()
    code.writeline("    # Calculate input window start positions")
    code.writeline("    h_start = h_out * stride_h - pad_h")
    code.writeline("    w_start = w_out * stride_w - pad_w")
    code.newline()
    code.writeline("    # Pre-calculate base input index")
    code.writeline("    base_input_idx = n * C * H * W + c * H * W")
    code.newline()
    code.writeline("    # Initialize accumulator and count")
    code.writeline("    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)")
    code.writeline("    count = tl.zeros([BLOCK_SIZE], dtype=tl.float32)")
    code.newline()
    code.writeline(f"    # Unrolled pooling window ({kernel_h}x{kernel_w})")

    # Generate unrolled code
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            if is_large_kernel:
                # Compact version for large kernels
                code.writeline(f"    # Position ({kh}, {kw})")
                code.writeline(f"    h_in = h_start + {kh}")
                code.writeline(f"    w_in = w_start + {kw}")
                code.writeline(
                    "    valid = ((h_in >= 0) & (h_in < H) & "
                    "(w_in >= 0) & (w_in < W)) & mask"
                )
                code.writeline(
                    "    acc = acc + tl.load(input_ptr + base_input_idx + "
                    "h_in * W + w_in, mask=valid, other=0.0)"
                )
                if count_include_pad:
                    code.writeline(
                        "    in_pad = ((h_in >= -pad_h) & (h_in < H + pad_h) & "
                        "(w_in >= -pad_w) & (w_in < W + pad_w)) & mask"
                    )
                    code.writeline("    count = count + tl.where(in_pad, 1.0, 0.0)")
                else:
                    code.writeline("    count = count + tl.where(valid, 1.0, 0.0)")
            else:
                # Full unroll for small kernels
                code.writeline(f"    # Position ({kh}, {kw})")
                code.writeline(f"    h_in_{kh}_{kw} = h_start + {kh}")
                code.writeline(f"    w_in_{kh}_{kw} = w_start + {kw}")
                code.newline()
                code.writeline(
                    f"    valid_h_{kh}_{kw} = (h_in_{kh}_{kw} >= 0) & "
                    f"(h_in_{kh}_{kw} < H)"
                )
                code.writeline(
                    f"    valid_w_{kh}_{kw} = (w_in_{kh}_{kw} >= 0) & "
                    f"(w_in_{kh}_{kw} < W)"
                )
                code.writeline(
                    f"    valid_{kh}_{kw} = valid_h_{kh}_{kw} & "
                    f"valid_w_{kh}_{kw} & mask"
                )
                code.newline()
                code.writeline(
                    f"    input_idx_{kh}_{kw} = base_input_idx + "
                    f"h_in_{kh}_{kw} * W + w_in_{kh}_{kw}"
                )
                code.writeline(
                    f"    input_val_{kh}_{kw} = tl.load(input_ptr + "
                    f"input_idx_{kh}_{kw}, mask=valid_{kh}_{kw}, other=0.0)"
                )
                code.newline()
                code.writeline(f"    acc = acc + input_val_{kh}_{kw}")

                if count_include_pad:
                    code.writeline(
                        f"    in_pad_h_{kh}_{kw} = (h_in_{kh}_{kw} >= -pad_h) & "
                        f"(h_in_{kh}_{kw} < H + pad_h)"
                    )
                    code.writeline(
                        f"    in_pad_w_{kh}_{kw} = (w_in_{kh}_{kw} >= -pad_w) & "
                        f"(w_in_{kh}_{kw} < W + pad_w)"
                    )
                    code.writeline(
                        f"    in_window_{kh}_{kw} = in_pad_h_{kh}_{kw} & "
                        f"in_pad_w_{kh}_{kw} & mask"
                    )
                    code.writeline(
                        f"    count = count + tl.where(in_window_{kh}_{kw}, 1.0, 0.0)"
                    )
                else:
                    code.writeline(
                        f"    count = count + tl.where(valid_{kh}_{kw}, 1.0, 0.0)"
                    )

    code.newline()
    code.writeline("    # Apply divisor override if provided")
    code.writeline("    if divisor_override > 0:")
    code.writeline(
        "        divisor = tl.full([BLOCK_SIZE], divisor_override, dtype=tl.float32)"
    )
    code.writeline("    else:")
    code.writeline("        divisor = count")
    code.newline()
    code.writeline("    # Avoid division by zero")
    code.writeline("    divisor = tl.where(divisor > 0, divisor, 1.0)")
    code.newline()
    code.writeline("    # Calculate average")
    code.writeline("    output = acc / divisor")
    code.newline()
    code.writeline("    # Store result")
    code.writeline("    tl.store(output_ptr + idx, output, mask=mask)")

    return code


def get_or_create_codegen_kernel(kernel_h, kernel_w, count_include_pad):
    """
    Get cached kernel or generate new one.

    Args:
        kernel_h: Kernel height
        kernel_w: Kernel width
        count_include_pad: Whether to include padding in count

    Returns:
        Compiled Triton kernel function
    """
    # Create cache key
    cache_key = (kernel_h, kernel_w, count_include_pad)

    if cache_key in _codegen_kernel_cache:
        return _codegen_kernel_cache[cache_key]

    # Generate kernel code using IndentedBuffer
    code = IndentedBuffer()
    code = generate_avg_pool2d_kernel_code(kernel_h, kernel_w, count_include_pad, code)

    # Write to file in cache directory
    file_name = f"avg_pool2d_{kernel_h}x{kernel_w}_pad{int(count_include_pad)}.py"
    file_path = code_cache_dir() / file_name
    write_atomic(file_path, code.getvalue())

    # Load the module from file
    kernel_name = f"avg_pool2d_kernel_{kernel_h}x{kernel_w}_pad{int(count_include_pad)}"
    module_name = f"_gen_avg_pool2d_{kernel_h}x{kernel_w}_pad{int(count_include_pad)}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    kernel_func = getattr(m, kernel_name)

    # Cache the kernel
    _codegen_kernel_cache[cache_key] = kernel_func

    logger.debug(
        f"Generated and cached kernel for {kernel_h}x{kernel_w}, "
        f"count_include_pad={count_include_pad}"
    )

    return kernel_func


def avg_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    """
    Average pooling operation using dynamic code generation.

    Generates optimized Triton kernels at runtime based on kernel size.
    Uses adaptive strategy: full loop unrolling for small kernels (<=4x4),
    compact code with variable reuse for large kernels (>4x4).

    Args:
        input: Input tensor [N, C, H, W]
        kernel_size: Size of pooling kernel (int or tuple)
        stride: Stride of pooling operation (default: same as kernel_size)
        padding: Padding to add to input (int or tuple)
        ceil_mode: When True, use ceil instead of floor for output size calculation
        count_include_pad: When True, include padding in average calculation
        divisor_override: If specified, use this value as divisor instead of pool region size

    Returns:
        Output tensor after average pooling [N, C, H_out, W_out]
    """
    logger.debug("GEMS AVG_POOL2D")

    # Handle kernel_size
    if isinstance(kernel_size, int):
        kernel_h = kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size

    # Handle stride
    if stride is None:
        stride_h = kernel_h
        stride_w = kernel_w
    elif isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    # Handle padding
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    # Get input dimensions
    N, C, H, W = input.shape

    # Calculate output dimensions
    if ceil_mode:
        H_out = (H + 2 * pad_h - kernel_h + stride_h - 1) // stride_h + 1
        W_out = (W + 2 * pad_w - kernel_w + stride_w - 1) // stride_w + 1
    else:
        H_out = (H + 2 * pad_h - kernel_h) // stride_h + 1
        W_out = (W + 2 * pad_w - kernel_w) // stride_w + 1

    # Create output tensor
    output = torch.empty((N, C, H_out, W_out), device=input.device, dtype=input.dtype)

    # Handle divisor_override
    if divisor_override is None:
        divisor_override = -1

    # Get or generate kernel
    kernel_func = get_or_create_codegen_kernel(kernel_h, kernel_w, count_include_pad)

    # Launch kernel with autotune
    grid = lambda meta: (triton.cdiv(N * C * H_out * W_out, meta["BLOCK_SIZE"]),)

    kernel_func[grid](
        input,
        output,
        N,
        C,
        H,
        W,
        H_out,
        W_out,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        divisor_override,
    )

    return output


def avg_pool2d_out(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
    *,
    out,
):
    """
    Average pooling operation with output tensor specified.

    Same as avg_pool2d but writes result to pre-allocated output tensor.

    Args:
        input: Input tensor [N, C, H, W]
        kernel_size: Size of pooling kernel (int or tuple)
        stride: Stride of pooling operation (default: same as kernel_size)
        padding: Padding to add to input (int or tuple)
        ceil_mode: When True, use ceil instead of floor for output size calculation
        count_include_pad: When True, include padding in average calculation
        divisor_override: If specified, use this value as divisor instead of pool region size
        out: Output tensor to write result to

    Returns:
        The output tensor (same as out parameter)
    """
    result = avg_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    out.copy_(result)
    return out
