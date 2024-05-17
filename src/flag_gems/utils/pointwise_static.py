from itertools import chain
import importlib
from typing import List, Callable, Mapping
import hashlib

import torch
import triton
from triton.runtime.jit import JITFunction

from flag_gems.utils.shape_utils import (
    broadcast_shapes,
    broadcasted_stride,
    c_contiguous_stride,
    volume,
    Shape,
    Stride,
)
from flag_gems.utils.code_cache import cache_dir
from flag_gems.utils.inliner import inline_function
from flag_gems.utils.code_utils import IndentedBuffer, NameSpace


def generate_pointwise_wrapper(
    inputs: List[torch.Tensor],
    num_outputs: int,
    wrapper_name: str,
    kernel_name: str,
    scalar_fn: JITFunction,
    code: IndentedBuffer,
) -> IndentedBuffer:
    """Generate code to call kernel for static shape.
    Shape & stride computations are done in code-generation time.
    """
    # number of inputs
    num_inputs = len(inputs)

    # compute task index space from input shapes
    tensor_shapes = tuple(
        item.shape
        for item in chain(
            inputs,
        )
    )
    shape = broadcast_shapes(tensor_shapes)
    num_tasks = volume(shape)

    # # input strides for each input tensor w.r.t. the task index space
    # input_strides = tuple(broadcasted_stride(item.shape, item.stride(), shape) for item in inputs)
    # # outputs are all c-contiguous, not the best actually
    # output_strides = tuple(c_contiguous_stride(shape) for _ in range(num_outputs))

    # task partitioning, 1d task indexing
    tile_size = 512
    num_warps = 4
    grid = triton.cdiv(num_tasks, tile_size), 1, 1

    # wrapper signature
    input_parameters: List[str] = [f"in{i}: torch.Tensor" for i in range(num_inputs)]
    arguments: str = ", ".join(input_parameters)
    wrapper_signature: str = f"def {wrapper_name}({arguments}):"
    code.writeline(wrapper_signature)

    with code.indent():
        # docstring
        wrapper_docstring: str = f'"""Generated pointwise kernel with {num_inputs} input tensors and {num_outputs} output tensors."""'
        code.writeline(wrapper_docstring)

        # ----- output allocation -----
        # NOTE: the layout of the output depends on
        # 1. the first input, if it has no internal overlapping and has the same shape as the output, the output follows its layout
        # 2. otherwise, the output is C-contiguous
        shape = broadcast_shapes([item.shape for item in inputs])
        code.writeline(f"shape = {shape}")

        code.writeline("if shape == in0.shape:")
        with code.indent():
            for i in range(num_outputs):
                allocate_output: str = f"out{i}: torch.Tensor = torch.empty_like(in0)"
                code.writeline(allocate_output)
        code.writeline("else:")
        with code.indent():
            for i in range(num_outputs):
                allocate_output: str = f"out{i}: torch.Tensor = torch.empty(shape, dtype=in0.dtype, device=in0.device)"
                code.writeline(allocate_output)

        # input strides for each input tensor w.r.t. the task index space
        input_strides = tuple(
            broadcasted_stride(item.shape, item.stride(), shape) for item in (inputs)
        )
        # outputs are all c-contiguous, not the best actually
        output_strides = tuple(c_contiguous_stride(shape) for _ in range(num_outputs))

        # grid
        grid_stmt: str = f"grid = {grid}"
        code.writeline(grid_stmt)

        # launch kernel
        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)
        with code.indent():
            # input tensors
            input_args: str = ", ".join(f"in{i}" for i in range(num_inputs))
            code.writeline(f"{input_args}, # input tensors")

            # output tensors
            output_args: str = ", ".join(f"out{i}" for i in range(num_outputs))
            code.writeline(f"{output_args}, # output tensors")

            code.writeline(f"tile_size={tile_size},")
            code.writeline(f"num_warps={num_warps},")
        code.writeline(")")

        # return
        code.writeline(f"return {output_args}")
        code.newline()

    # generate triton kernel
    code = generate_pointwise_kernel(
        input_strides, output_strides, shape, "_jit_function", scalar_fn, code
    )
    return code


def generate_pointwise_kernel(
    input_strides: List[Stride],
    output_strides: List[Stride],
    task_space: Shape,
    kernel_name: str,
    scalar_fn: JITFunction,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")

    num_inputs = len(input_strides)
    num_outputs = len(output_strides)
    rank = len(task_space)

    function_ns = NameSpace()

    # signature
    with code.indent():
        input_parameters = [f"in{i}_ptr" for i in range(num_inputs)]
        output_parameters = [f"out{i}_ptr" for i in range(num_outputs)]
        ptr_arguments = ", ".join(chain(input_parameters, output_parameters))
        code.writeline(f"{ptr_arguments},")
        for arg_name in ptr_arguments:
            function_ns.create_name(arg_name)

        code.writeline("tile_size: tl.constexpr,")
        function_ns.create_name("tile_size")

    code.writeline("):")

    num_tasks = volume(task_space)
    with code.indent():
        # get pid
        code.writeline("# task id & masking")
        pid_stmt = "pid = tl.program_id(0)"
        code.writeline(pid_stmt)
        function_ns.create_name("pid")
        # tile size
        tid_stmt = "tid = pid * tile_size + tl.arange(0, tile_size)"
        code.writeline(tid_stmt)
        function_ns.create_name("tid")
        # masking
        mask_stmt: str = f"mask = tid < {num_tasks}"
        code.writeline(mask_stmt)
        function_ns.create_name("mask")
        code.newline()

        # reconstruct multi index
        code.writeline("# multi index recontruction")
        for i in reversed(range(rank)):
            code.writeline(f"i{i} = tid % {task_space[i]}")
            code.writeline(f"i{i}")
            if i > 0:
                code.writeline(f"tid //= {task_space[i]}")
        code.newline()

        # loads
        code.writeline("# loads")
        for i in range(num_inputs):
            ptrs_expr: str = " + ".join(
                f"i{j} * {input_strides[i][j]}" for j in range(rank)
            )
            ptrs_expr: str = f"in{i}_ptr + {ptrs_expr}"
            load_stmt: str = f"in{i} = tl.load({ptrs_expr}, mask=mask)"
            function_ns.create_name(f"in{i}")  # add to the namespace
            code.writeline(load_stmt)
        code.newline()

        # compute
        code.writeline("# compute")
        compute_body = inline_function(
            scalar_fn,
            [f"in{i}" for i in range(num_inputs)],
            [f"out{i}" for i in range(num_outputs)],
            function_ns,
        )
        for line in compute_body.strip().splitlines():
            code.writeline(line)
        code.newline()

        # loads
        code.writeline("# stores")
        for i in range(num_outputs):
            ptrs_expr: str = " + ".join(
                f"i{j} * {output_strides[i][j]}" for j in range(rank)
            )
            ptrs_expr: str = f"out{i}_ptr + {ptrs_expr}"
            load_stmt: str = f"tl.store({ptrs_expr}, out{i}, mask=mask)"
            code.writeline(load_stmt)
        code.newline()

    return code


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import math")
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("from triton import language as tl")
    code.newline()

    code.writeline("from flag_gems.utils import libentry")
    code.newline()
    return code


class PointwiseStaticFunction:
    """Utility to generate function for general pointwise operation. It generate wrapper & JITFunction
    which are specialized according to each input tensors shape & strides.
    The generated code are written out to the cache directory (defaults to ~/.flaggems).
    """

    def __init__(self, scalar_fn: JITFunction):
        self.scalar_fn = scalar_fn
        self.scalar_fn_cache_key = scalar_fn.cache_key
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = hashlib.sha256(
            f"{self.arg_key(*args, **kwargs)}".encode("utf-8")
        ).hexdigest()
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            # generate file & import it
            code = IndentedBuffer()
            code = generate_imports(code)
            code = generate_pointwise_wrapper(
                args, 1, "_wrapper", "_jit_function", self.scalar_fn, code
            )

            file_name = f"pointwise_static_{self.scalar_fn_cache_key}_spec_{key}.py"
            with open(cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())
                f.close()

            # load
            spec = importlib.util.spec_from_file_location("_add_module", f.name)
            m = importlib.util.module_from_spec(spec)
            # do not expose it to sys.modules
            # sys.modules["_add_module"] = m
            spec.loader.exec_module(m)
            overload = getattr(m, "_wrapper")
            self.overloads[key] = overload
        return overload(*args, **kwargs)

    def arg_key(self, *args, **kwargs):
        tensors = [item for item in args if torch.is_tensor(item)]
        shapes = tuple(item.shape for item in tensors)
        strides = tuple(item.stride() for item in tensors)
        return (shapes, strides)


def pointwise_static(function: JITFunction):
    return PointwiseStaticFunction(function)


if __name__ == "__main__":

    @pointwise_static
    @triton.jit
    def f(a, b):
        c = a + b
        return tl.sigmoid(c)

    a = torch.randn(4, 4, device="cuda")
    b = torch.randn_like(a)
    # print(a.shape, a.stride())

    print(f(a, b))
    print(torch.sigmoid(a + b))

    import triton

    t1 = triton.testing.do_bench(lambda: f(a, b), return_mode="median")
    t2 = triton.testing.do_bench(lambda: torch.sigmoid(a + b), return_mode="median")
    print(t1)
    print(t2)
