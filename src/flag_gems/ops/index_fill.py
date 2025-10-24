import importlib
import logging
import os
from typing import Any, Callable, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")
    code.newline()
    code.newline()
    return code


def generate_index_fill_kernel(
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # Decorators with autotune for 2D grid
    code.writeline("@libentry()")
    code.writeline("@triton.autotune(")
    with code.indent():
        code.writeline("configs=[")
        with code.indent():
            # Small indices, large elements per index
            code.writeline(
                "triton.Config({'BLOCK_M': 16, 'BLOCK_N': 1024}, num_warps=4, num_stages=2),"
            )
            code.writeline(
                "triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512}, num_warps=4, num_stages=2),"
            )
            # Balanced
            code.writeline(
                "triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512}, num_warps=4, num_stages=2),"
            )
            code.writeline(
                "triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=4, num_stages=2),"
            )
            code.writeline(
                "triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=4, num_stages=2),"
            )
            # Large indices, small elements per index
            code.writeline(
                "triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=1),"
            )
            code.writeline(
                "triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=1),"
            )
            # High warps for large tensors
            code.writeline(
                "triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512}, num_warps=8, num_stages=3),"
            )
            code.writeline(
                "triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=8, num_stages=2),"
            )
        code.writeline("],")
        code.writeline("key=['M', 'N', 'dim_stride'],")
    code.writeline(")")
    code.writeline("@triton.jit")

    # Signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("output_ptr,")
        code.writeline("index_ptr,")
        code.writeline("value,")
        code.writeline("dim_stride,")
        code.writeline("dim_size,")
        code.writeline("M,")
        code.writeline("N,")
        code.writeline("BLOCK_M: tl.constexpr,")
        code.writeline("BLOCK_N: tl.constexpr,")
    code.writeline("):")

    # Kernel body with 2D grid
    with code.indent():
        code.writeline("# 2D grid: M for indices, N for elements per index")
        code.writeline("pid_m = tle.program_id(axis=0)")
        code.writeline("pid_n = tle.program_id(axis=1)")
        code.writeline("")
        code.writeline("# Index dimension offsets")
        code.writeline("offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)")
        code.writeline("mask_m = offset_m < M")
        code.writeline("")
        code.writeline("# Load indices")
        code.writeline("idx = tl.load(index_ptr + offset_m, mask=mask_m, other=0)")
        code.writeline("# Handle negative indices (wrap to positive)")
        code.writeline("idx = tl.where(idx < 0, idx + dim_size, idx)")
        code.writeline("")
        code.writeline("# Elements dimension offsets")
        code.writeline("offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)")
        code.writeline("mask_n = offset_n < N")
        code.writeline("")

        # Optimized position calculation
        code.writeline("# Broadcast to 2D")
        code.writeline("idx_2d = idx[:, None]")
        code.writeline("offset_n_2d = offset_n[None, :]")
        code.writeline("")
        code.writeline("# Calculate position: (outer * size + idx) * stride + inner")
        code.writeline("outer_offset = offset_n_2d // dim_stride")
        code.writeline("inner_offset = offset_n_2d % dim_stride")
        code.writeline(
            "positions = (outer_offset * dim_size + idx_2d) * dim_stride + inner_offset"
        )
        code.writeline("")
        code.writeline("# Combined mask")
        code.writeline("mask_2d = mask_m[:, None] & mask_n[None, :]")
        code.writeline("mask_2d = mask_2d & (idx_2d >= 0) & (idx_2d < dim_size)")
        code.writeline("")

        # Store the value
        code.writeline("tl.store(output_ptr + positions, value, mask=mask_2d)")

    code.newline()
    code.newline()
    return code


def generate_index_fill_wrapper(
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline(
        f"def {wrapper_name}(output, index, value, dim_stride, dim_size, M, N):"
    )

    with code.indent():
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline("triton.cdiv(M, meta['BLOCK_M']),")
            code.writeline("triton.cdiv(N, meta['BLOCK_N']),")
        code.writeline(")")
        code.writeline("")
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            code.writeline("output,")
            code.writeline("index,")
            code.writeline("value,")
            code.writeline("dim_stride,")
            code.writeline("dim_size,")
            code.writeline("M,")
            code.writeline("N,")
        code.writeline(")")
        code.writeline("return output")

    code.newline()
    code.newline()
    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: [output, index, value, dim_stride, dim_size, M, N]
    code = generate_imports(code)
    code = generate_index_fill_kernel(kernel_name, code)
    code = generate_index_fill_wrapper(wrapper_name, kernel_name, code)
    return code


class IndexFillFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[int, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = self.arg_key(*args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                args,
                "_index_fill_wrapper",
                "_index_fill_kernel",
                code,
            )

            file_name = f"index_fill_rank_{key}_pid_{self.pid}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            # Load the generated module
            spec = importlib.util.spec_from_file_location(
                f"_gen_index_fill_rank_{key}_pid_{self.pid}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_fill_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        # Key based on output tensor rank
        return args[0].ndim


_index_fill_func = IndexFillFunction()


def index_fill(input, dim, index, value):
    """
    Fill elements of input tensor along a dimension with a scalar value.

    Args:
        input: Input tensor
        dim: Dimension along which to fill
        index: 1-D tensor containing indices to fill
        value: Scalar value (Python number or 0-dim tensor) to fill with

    Returns:
        New tensor with filled values

    Raises:
        ValueError: If input/index not on CUDA or value is not a scalar
        RuntimeError: If index is not 1-D
        IndexError: If dim is out of range
    """
    logger.debug("GEMS INDEX FILL")

    # Validate inputs
    if not input.is_cuda:
        raise ValueError("input must be on CUDA device")
    if not index.is_cuda:
        raise ValueError("index must be on CUDA device")
    if index.ndim != 1:
        raise RuntimeError("index_fill(): Expected a 1-D tensor for index")

    # Convert negative dim to positive
    ndim = input.ndim
    original_dim = dim
    if dim < 0:
        dim = dim + ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], but got {original_dim})"
        )

    # Early return for empty index
    if index.numel() == 0:
        return input.clone()

    # Ensure index is int64
    if index.dtype != torch.int64:
        index = index.to(torch.int64)

    # Convert value to scalar if it's a tensor
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.item()
        else:
            raise ValueError("Value tensor must be a scalar (0-dim tensor or number)")

    # Clone and ensure contiguous in a single operation
    output = input.clone(memory_format=torch.contiguous_format)

    # Pre-compute dim-related information
    dim_stride = output.stride()[dim]
    dim_size = output.shape[dim]

    # Calculate grid dimensions for 2D parallelization
    M = index.numel()  # Number of indices to process
    N = output.numel() // dim_size  # Elements per index (contiguous tensor assumed)

    # Call the code-generated function
    _index_fill_func(output, index, value, dim_stride, dim_size, M, N)

    return output
