import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, NameSpace


# --------------------------- padding wrapper genration -----------------------------------
def parameter_for_wrapper() -> str:
    """Generate parameter declaration with type annotation for wrapper function.
    Example: in0: torch.Tensor, val0: float, out0: torch.Tensor
    """
    parameters: List[str] = []

    parameters.append("in0")
    parameters.append("pad")
    parameters.append("mode")
    parameters.append("value=0")
    return ", ".join(parameters)


def parameter_for_wrapper_out() -> str:
    """Generate parameter declaration with type annotation for wrapper function.
    Example: in0: torch.Tensor, val0: float, out0: torch.Tensor
    """
    parameters: List[str] = []

    parameters.append("in0")
    parameters.append("out0")
    parameters.append("dst_shape")
    parameters.append("pad_before")
    parameters.append("pad_after")
    parameters.append("mode")
    parameters.append("value=0")

    return ", ".join(parameters)


def parameter_ref_for_wrapper() -> str:
    """Generate parameter reference for wrapper function.
    Example: in0, val0, out0, out0_offset
    """
    parameters: List[str] = []

    parameters.append("in0")
    parameters.append("out0")
    parameters.append("dst_shape")
    parameters.append("pad_before")
    parameters.append("pad_after")
    parameters.append("mode")
    parameters.append("value")

    return ", ".join(parameters)


def output_ref_for_wrapper() -> str:
    return "out0"


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import math")
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("from triton import language as tl")
    code.newline()
    code.writeline("from flag_gems.utils.libentry import libentry")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")
    code.writeline("from flag_gems.utils.type_utils import type_promotion")
    code.newline()
    code.newline()
    return code


def generate_functional_padding_wrapper(
    wrapper_name: str,
    destination_passing_func_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # wrapper signature
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("ndim = in0.ndim")
        code.writeline("pad_size = len(pad)")
        code.writeline("assert pad_size % 2 == 0")
        code.newline()
        code.writeline("pad_before = [0 for _ in range(ndim)]")
        code.writeline("pad_after = [0 for _ in range(ndim)]")
        code.newline()
        code.writeline("pad_pair = pad_size // 2 ")
        code.writeline("for i in range(pad_pair): ")
        with code.indent():
            code.writeline("pad_before[ndim - i - 1] = pad[2 * i]")
            code.writeline("pad_after[ndim - i - 1] = pad[2 * i + 1]")
        code.writeline("dst_shape = list(in0.shape)")
        code.writeline("for i in range(ndim): ")
        with code.indent():
            code.writeline("dst_shape[i] += pad_before[i] + pad_after[i]")

        code.writeline(
            ("out0 = torch.empty(dst_shape, device=in0.device, dtype=in0.dtype)")
        )

        # call destination_passing_func
        output_names: str = output_ref_for_wrapper()
        call_str = (
            f"{output_names} = {destination_passing_func_name}"
            f"({parameter_ref_for_wrapper()})"
        )
        code.writeline(call_str)

        return_str = "return out0"
        code.writeline(return_str)
        code.newline()
        code.newline()

    return code


def generate_destination_passing_padding_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # wrapper signature
    parameters: str = parameter_for_wrapper_out()

    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        # docstring
        code.writeline("BLOCK_SIZE = 256")
        code.writeline("grid = (triton.cdiv(out0.numel(), BLOCK_SIZE), 1, 1)")
        code.newline()

        code.writeline("x_shape = in0.shape")
        code.writeline("in_strides0 = in0.stride()")
        code.writeline("out_strides = out0.stride()")

        # input strides for each input tensor w.r.t. the task index space
        if rank > 0:
            code.writeline("# strides of each tensor argument w.r.t the task space")
            for i in range(rank):
                code.writeline(f"valid_dim{i}_start = pad_before[{i}]")

                code.writeline(f"valid_dim{i}_end = dst_shape[{i}] - pad_after[{i}]")

        code.newline()

        code.writeline("IS_CONSTANT = mode == 'constant'")
        code.writeline("IS_REFLECT = mode == 'reflect'")
        code.writeline("IS_REPLICATE = mode == 'replicate'")
        code.writeline("IS_CIRCULAR = mode == 'circular'")

        code.newline()

        # grid
        code.writeline("# kernel launch")

        # launch kernel
        code.writeline("with torch.cuda.device(in0.device):")
        with code.indent():
            kernel_launch: str = f"{kernel_name}[grid]("
            code.writeline(kernel_launch)

            with code.indent():
                code.writeline("in0, out0, ")

                if rank > 0:
                    s = ", ".join(f"x_shape[{j}]" for j in range(rank))
                    code.writeline(f"{s}, # shape for x")

                    s = ", ".join(f"in_strides0[{j}]" for j in range(rank))
                    code.writeline(f"{s}, # stride for x")

                    s = ", ".join(f"out_strides[{j}]" for j in range(rank))
                    code.writeline(f"{s}, # stride for out")

                    s = ", ".join(f"valid_dim{j}_start" for j in range(rank))
                    code.writeline(f"{s}, # valid dim start")

                    s = ", ".join(f"valid_dim{j}_end" for j in range(rank))
                    code.writeline(f"{s}, # valid dim end")

                    code.writeline("in0.numel(), ")
                    code.writeline("out0.numel(), ")
                    code.writeline("value, ")
                    code.writeline("IS_CONSTANT, ")
                    code.writeline("IS_REFLECT, ")
                    code.writeline("IS_REPLICATE, ")
                    code.writeline("IS_CIRCULAR, ")
                    code.writeline("BLOCK_SIZE, ")
            code.writeline(")")

        code.writeline("return out0")
        code.newline()
        code.newline()
    return code


def generate_pad_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # make the inlined function visible in the context
    code.newline()

    # the decorators
    code.writeline("@libentry()")
    non_specialize_arg_names = ["value"]
    code.writeline(f"@triton.jit(do_not_specialize={non_specialize_arg_names})")

    # signature
    code.writeline(f"def {kernel_name}(")
    function_ns = NameSpace()
    with code.indent():
        code.writeline("in0_ptr: tl.tensor, # of tl.pointer_type")
        function_ns.create_name("in0_ptr")

        code.writeline("out0_ptr: tl.tensor, # of tl.pointer_type")
        function_ns.create_name("out0_ptr")

        if rank > 0:
            # shape for inputs
            for j in range(rank):
                function_ns.create_name(f"x_shape{j}")
            shape_args = ", ".join(f"x_shape{j}: int" for j in range(rank))
            code.writeline(f"{shape_args}, # shape for x")

            # shape for inputs
            for j in range(rank):
                function_ns.create_name(f"in_strides{j}")
            stride_args = ", ".join(f"in_strides{j}: int" for j in range(rank))
            code.writeline(f"{stride_args}, # stride for x")

            # shape for inputs
            for j in range(rank):
                function_ns.create_name(f"out_strides{j}")
            stride_args = ", ".join(f"out_strides{j}: int" for j in range(rank))
            code.writeline(f"{stride_args}, # stride for out")

            # shape for inputs
            for j in range(rank):
                function_ns.create_name(f"valid_dim{j}_start")
            stride_args = ", ".join(f"valid_dim{j}_start: int" for j in range(rank))
            code.writeline(f"{stride_args}, # valid dim start")

            # shape for inputs
            for j in range(rank):
                function_ns.create_name(f"valid_dim{j}_end")
            stride_args = ", ".join(f"valid_dim{j}_end: int" for j in range(rank))
            code.writeline(f"{stride_args}, # valid dim end")

            code.writeline("in_elem_cnt: tl.constexpr, ")
            code.writeline("out_elem_cnt: tl.constexpr, ")
            code.writeline("value, # padding value")
            code.writeline("IS_CONSTANT: tl.constexpr, ")
            code.writeline("IS_REFLECT: tl.constexpr, ")
            code.writeline("IS_REPLICATE: tl.constexpr, ")
            code.writeline("IS_CIRCULAR: tl.constexpr, ")
            code.writeline("BLOCK_SIZE: tl.constexpr, ")

    code.writeline("):")

    with code.indent():
        code.writeline("pid = tle.program_id(0)")
        code.writeline("block_offset = pid * BLOCK_SIZE")
        code.writeline("offset = block_offset + tl.arange(0, BLOCK_SIZE)")
        code.newline()

        code.writeline("remaining = offset ")
        for i in range(rank):
            code.writeline(f"idx = remaining // out_strides{i}")
            code.writeline(f"dst_index_{i} = idx")
            code.writeline(f"remaining = remaining - idx * out_strides{i}")
            code.newline()

        code.writeline("if_pad_false_mask = tl.zeros((BLOCK_SIZE, ), dtype=tl.int32)")
        code.writeline("if_pad_true_mask = tl.full((BLOCK_SIZE, ), 1, dtype=tl.int32)")

        code.writeline(
            "cond = (dst_index_0 >= valid_dim0_start and dst_index_0 < valid_dim0_end) "
        )

        for i in range(1, rank):
            code.writeline(
                f"cond &= (dst_index_{i} >= valid_dim{i}_start and dst_index_{i} < valid_dim{i}_end)"
            )

        code.writeline(
            "if_pad = tl.where(cond, if_pad_false_mask, if_pad_true_mask).to(tl.int1)"
        )

        for i in range(rank):
            code.writeline(f"src_index_{i} = dst_index_{i} - valid_dim{i}_start ")

        for i in range(rank):
            code.writeline(
                f"src_index_{i} = tl.where(src_index_{i} < 0, 0, src_index_{i})"
            )

        code.newline()
        code.writeline("if IS_REFLECT: ")
        with code.indent():
            for i in range(rank):
                code.writeline(
                    f"""src_index_{i} = tl.where(dst_index_{i} < valid_dim{i}_start,
                        valid_dim{i}_start - dst_index_{i}, src_index_{i})"""
                )
            for i in range(rank):
                code.writeline(
                    f"""src_index_{i} = tl.where(dst_index_{i} >= valid_dim{i}_end,
                    (x_shape{i} + valid_dim{i}_start - 1) * 2 - dst_index_{i} - valid_dim{i}_start, src_index_{i})"""
                )

        code.newline()
        code.writeline("if IS_REPLICATE: ")
        with code.indent():
            for i in range(rank):
                code.writeline(
                    f"src_index_{i} = tl.where(dst_index_{i} < valid_dim{i}_start, 0, src_index_{i})"
                )
            for i in range(rank):
                code.writeline(
                    f"src_index_{i} = tl.where(dst_index_{i} >= valid_dim{i}_end, x_shape{i} - 1, src_index_{i})"
                )

        code.newline()
        code.writeline("if IS_CIRCULAR: ")
        with code.indent():
            for i in range(rank):
                code.writeline(
                    f"""src_index_{i} = tl.where(dst_index_{i} < valid_dim{i}_start,
                        dst_index_{i} + x_shape{i} - valid_dim{i}_start, src_index_{i})"""
                )
            for i in range(rank):
                code.writeline(
                    f"""src_index_{i} = tl.where(dst_index_{i} >= valid_dim{i}_end,
                        dst_index_{i} - valid_dim{i}_end, src_index_{i})"""
                )

        code.newline()

        code.writeline("src_offset = src_index_0 * in_strides0")
        for i in range(1, rank):
            code.writeline(f"src_offset += src_index_{i} * in_strides{i}")

        code.writeline(f"load_cond = src_index_{i} < x_shape{i}")
        for i in range(1, rank):
            code.writeline(f"load_cond &= src_index_{i} < x_shape{i}")

        code.writeline("if IS_CONSTANT: ")
        with code.indent():
            code.writeline(
                "x_val = tl.load(in0_ptr + src_offset, mask=(not if_pad) and load_cond, other=value)"
            )
        code.writeline("else: ")
        with code.indent():
            code.writeline(
                "x_val = tl.load(in0_ptr + src_offset, mask=load_cond, other=0)"
            )
        code.writeline("tl.store(out0_ptr + offset, x_val, mask=offset < out_elem_cnt)")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    destination_passing_func_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    shape = inputs[0].shape
    rank = len(shape)

    # the only runtime determined factor is the rank of the task space
    code = generate_imports(code)
    code = generate_functional_padding_wrapper(
        wrapper_name, destination_passing_func_name, code
    )
    code = generate_destination_passing_padding_wrapper(
        rank, destination_passing_func_name, kernel_name, code
    )
    code = generate_pad_kernel(rank, kernel_name, code)
    return code


class PadFunction:
    """Utility to generate function for general pointwise operation. It generate wrapper & JITFunction
    which are specialized according to the rank of the task space(the broadcasted shape of all input tensors).
    The generated code are written out to the cache directory (defaults to ~/.flaggems).
    """

    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        # note: kwargs should not be used in JITFunction directly
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            # generate file & import it
            code = IndentedBuffer()
            code = generate_code(
                args,
                "_wrapper",
                "_wrapper_out",
                "_jit_function",
                code,
            )

            file_name = f"constant_pad_rank_{key}_pid_{self.pid}.py"

            with open(cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            # do not expose it to sys.modules
            # sys.modules["_add_module"] = m
            spec.loader.exec_module(m)
            overload = getattr(m, "_wrapper")
            self.overloads[key] = overload
        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_pad_func = PadFunction()


def pad(self, pad, mode="constant", value=None):
    logging.debug("GEMS CONSTANT PAD ND")

    ndim = self.ndim

    if value is None:
        value = 0.0

    if mode == "reflect":
        ndim //= 2
        assert (
            len(pad) == 2 * ndim
        ), f"padding size is expected to be {2 * ndim}, but got {len(pad)}"

        for i in range(ndim):
            pad_l, pad_r = pad[2 * i], pad[2 * i + 1]
            input_l, input_r = (
                self.shape[ndim - (2 * i + 1) - 1],
                self.shape[ndim - (2 * i + 1)],
            )
            assert (
                pad_l < input_l and pad_r < input_r
            ), f"padding size should be less than the corresponding input dimension, \
                 but got padding size: {pad_l}, {pad_r}, input size: {self.shape}"

    if mode == "circular":
        ndim //= 2
        assert (
            len(pad) == 2 * ndim
        ), f"padding size is expected to be {2 * ndim}, but got {len(pad)}"
        for i in range(ndim):
            pad_l, pad_r = pad[2 * i], pad[2 * i + 1]
            input_size = self.shape[ndim - i - 1]
            assert (
                pad_l <= input_size and pad_r <= input_size
            ), "Padding value causes wrapping around more than once."

    out = _pad_func(self, pad, mode, float(value))
    return out
