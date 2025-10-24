"""
Dynamic code generation version of avg_pool2d for small kernels (2x2, 3x3)
This version unrolls the pooling loops at compile time for better performance.
"""

import hashlib
import importlib.util
import logging
import os
import tempfile

import torch
import triton

logger = logging.getLogger(__name__)

# Cache for dynamically generated kernels
_codegen_kernel_cache = {}


def generate_avg_pool2d_kernel_code(kernel_h, kernel_w, count_include_pad):
    """
    Generate Triton kernel source code with unrolled loops for specific kernel size.

    Adaptive strategy:
    - Small kernels (<=4x4): Full unroll with unique variables for better ILP
    - Large kernels (>4x4): Compact code with reused variables to reduce register pressure

    Args:
        kernel_h: Kernel height
        kernel_w: Kernel width
        count_include_pad: Whether to include padding in count

    Returns:
        str: Generated kernel source code
    """

    # Determine if this is a large kernel
    is_large_kernel = (kernel_h * kernel_w) > 16  # More than 4x4

    # Generate unrolled accumulation and counting code
    unrolled_code_lines = []

    for kh in range(kernel_h):
        for kw in range(kernel_w):
            if is_large_kernel:
                # Compact version for large kernels - reuse variables
                unrolled_code_lines.append(
                    f"""
    # Position ({kh}, {kw})
    h_in = h_start + {kh}
    w_in = w_start + {kw}
    valid = ((h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)) & mask
    acc = acc + tl.load(input_ptr + base_input_idx + h_in * W + w_in, mask=valid, other=0.0)
"""
                )
                if count_include_pad:
                    unrolled_code_lines.append(
                        """    in_pad = ((h_in >= -pad_h) & (h_in < H + pad_h) & """
                        """(w_in >= -pad_w) & (w_in < W + pad_w)) & mask
    count = count + tl.where(in_pad, 1.0, 0.0)
"""
                    )
                else:
                    unrolled_code_lines.append(
                        """    count = count + tl.where(valid, 1.0, 0.0)
"""
                    )
            else:
                # Full unroll for small kernels - unique variables for ILP
                unrolled_code_lines.append(
                    f"""
    # Position ({kh}, {kw})
    h_in_{kh}_{kw} = h_start + {kh}
    w_in_{kh}_{kw} = w_start + {kw}

    valid_h_{kh}_{kw} = (h_in_{kh}_{kw} >= 0) & (h_in_{kh}_{kw} < H)
    valid_w_{kh}_{kw} = (w_in_{kh}_{kw} >= 0) & (w_in_{kh}_{kw} < W)
    valid_{kh}_{kw} = valid_h_{kh}_{kw} & valid_w_{kh}_{kw} & mask

    input_idx_{kh}_{kw} = base_input_idx + h_in_{kh}_{kw} * W + w_in_{kh}_{kw}
    input_val_{kh}_{kw} = tl.load(input_ptr + input_idx_{kh}_{kw}, mask=valid_{kh}_{kw}, other=0.0)

    acc = acc + input_val_{kh}_{kw}
"""
                )

                # Add counting code based on count_include_pad
                if count_include_pad:
                    unrolled_code_lines.append(
                        f"""    in_pad_h_{kh}_{kw} = (h_in_{kh}_{kw} >= -pad_h) & (h_in_{kh}_{kw} < H + pad_h)
    in_pad_w_{kh}_{kw} = (w_in_{kh}_{kw} >= -pad_w) & (w_in_{kh}_{kw} < W + pad_w)
    in_window_{kh}_{kw} = in_pad_h_{kh}_{kw} & in_pad_w_{kh}_{kw} & mask
    count = count + tl.where(in_window_{kh}_{kw}, 1.0, 0.0)
"""
                    )
                else:
                    unrolled_code_lines.append(
                        f"""    count = count + tl.where(valid_{kh}_{kw}, 1.0, 0.0)
"""
                    )

    unrolled_code = "".join(unrolled_code_lines)

    # Generate complete kernel code
    kernel_code = f"""
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({{"BLOCK_SIZE": 128}}, num_warps=4),
        triton.Config({{"BLOCK_SIZE": 256}}, num_warps=4),
        triton.Config({{"BLOCK_SIZE": 512}}, num_warps=8),
        triton.Config({{"BLOCK_SIZE": 1024}}, num_warps=8),
    ],
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def avg_pool2d_kernel_{kernel_h}x{kernel_w}_pad{int(count_include_pad)}(
    input_ptr,
    output_ptr,
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
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)

    # Calculate indices for this block
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N * C * H_out * W_out

    # Convert flat index to (n, c, h_out, w_out)
    w_out = idx % W_out
    h_out = (idx // W_out) % H_out
    c = (idx // (W_out * H_out)) % C
    n = idx // (C * H_out * W_out)

    # Calculate input window start positions
    h_start = h_out * stride_h - pad_h
    w_start = w_out * stride_w - pad_w

    # Pre-calculate base input index
    base_input_idx = n * C * H * W + c * H * W

    # Initialize accumulator and count
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Unrolled pooling window ({kernel_h}x{kernel_w})
{unrolled_code}

    # Apply divisor override if provided
    if divisor_override > 0:
        divisor = tl.full([BLOCK_SIZE], divisor_override, dtype=tl.float32)
    else:
        divisor = count

    # Avoid division by zero
    divisor = tl.where(divisor > 0, divisor, 1.0)

    # Calculate average
    output = acc / divisor

    # Store result
    tl.store(output_ptr + idx, output, mask=mask)
"""

    return kernel_code


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

    # Generate kernel code
    kernel_code = generate_avg_pool2d_kernel_code(kernel_h, kernel_w, count_include_pad)

    # Create temporary file for the generated kernel
    code_hash = hashlib.md5(kernel_code.encode()).hexdigest()[:8]
    temp_dir = tempfile.gettempdir()
    kernel_file = os.path.join(
        temp_dir,
        f"avg_pool2d_{kernel_h}x{kernel_w}_pad{int(count_include_pad)}_{code_hash}.py",
    )

    # Write kernel code to file
    with open(kernel_file, "w") as f:
        f.write(kernel_code)

    # Dynamically import the kernel
    spec = importlib.util.spec_from_file_location(f"kernel_{code_hash}", kernel_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the kernel function
    kernel_name = f"avg_pool2d_kernel_{kernel_h}x{kernel_w}_pad{int(count_include_pad)}"
    kernel_func = getattr(module, kernel_name)

    # Cache the kernel
    _codegen_kernel_cache[cache_key] = kernel_func

    logger.debug(
        f"Generated and cached kernel for {kernel_h}x{kernel_w}, count_include_pad={count_include_pad}"
    )

    return kernel_func


def avg_pool2d_codegen(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    """
    Code-generated version of avg_pool2d for small kernels.

    Args:
        input: Input tensor [N, C, H, W]
        kernel_size: Size of pooling kernel
        stride: Stride of pooling operation
        padding: Padding to add to input
        ceil_mode: When True, use ceil instead of floor for output size
        count_include_pad: When True, include padding in average calculation
        divisor_override: If specified, override divisor for averaging

    Returns:
        Output tensor after avg pooling
    """
    logger.debug("GEMS AVG_POOL2D (CodeGen)")

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
    Average pooling using pure code generation with unrolled loops.
    Works for all kernel sizes.
    """
    return avg_pool2d_codegen(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
