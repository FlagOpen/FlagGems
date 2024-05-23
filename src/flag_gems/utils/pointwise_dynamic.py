from itertools import chain
import importlib
from typing import List, Callable, Mapping, Optional
from functools import partial
import torch
import triton
from triton.runtime.jit import JITFunction
from triton import language as tl

from flag_gems.utils.shape_utils import (
    broadcast_shapes,
    Shape,
)


from flag_gems.utils.code_cache import cache_dir
from flag_gems.utils.inliner import inline_function
from flag_gems.utils.code_utils import IndentedBuffer, NameSpace


from dataclasses import dataclass
from triton.runtime.jit import JITFunction
import torch
from enum import Enum


# ------------------ Operation Description ---------------------------
class Intent(Enum):
    IN = 0
    OUT = 1

    def __str__(self):
        return self.name.lower()

def type_name(type) -> str:
    "Render typename as string, work for both (bool, int, float, str) and torch.dtype object"
    if type in (bool, int, float, str):
        return type.__name__
    if isinstance(type, torch.dtype):
        return str(type)
    return str(type)

@dataclass
class OPDesc:
    is_tensor: List[bool]
    intents: List[Intent]
    dtypes: List[torch.dtype]
    scalar_fn: JITFunction

    def __post_init__(self):
        assert len(self.is_tensor) == len(self.intents) == len(self.dtypes)
        for i in range(0, len(self.is_tensor)):
            if not self.is_tensor[i]:
                self.intents[i] = Intent.IN # non tensors can only be input now


    def arity(self):
        return len(self.is_tensor)

    def num_input_tensors(self):
        n = 0
        for i in range(0, len(self.is_tensor)):
            if self.is_tensor[i] and self.intents[i] == Intent.IN:
                n += 1
        return n

    def num_output_tensors(self):
        n = 0
        for i in range(0, len(self.is_tensor)):
            if self.is_tensor[i] and self.intents[i] == Intent.OUT:
                n += 1
        return n

    def num_non_tensor_args(self):
        n = 0
        for i in range(0, len(self.is_tensor)):
            if (not self.is_tensor[i]) and self.intents[i] == Intent.IN:
                n += 1
        return n

    def __str__(self):
        repr = f"OPDesc(\n\tis_tensor: {self.is_tensor}\n\tintents: {self.intents}\n\tdtypes: {[type_name(t) for t in self.dtypes]}\n\tshapes: {self.shapes}\n)"
        return repr


def create_description(
    is_tensor: List[bool],
    intents: List[Intent],
    dtypes: List,
    scalar_fn: JITFunction,
):
    return OPDesc(is_tensor, intents, dtypes, scalar_fn)


# --------------------------- pointwise wrapper genration -----------------------------------
def parameter_for_wrapper(op_desc: OPDesc, include_outputs=False) -> str:
    """Generate parameter declaration with type annotation for wrapper function"""
    parameters: List[str] = []
    input_tensor_index = 0
    non_tensor_index = 0
    output_tensor_index = 0
    for i in range(op_desc.arity()):
        if op_desc.intents[i] == Intent.IN:
            if op_desc.is_tensor[i]:
                parameters.append(f"in{input_tensor_index}: torch.Tensor")
                input_tensor_index += 1
            else:
                if op_desc.dtypes[i] is not None:
                    parameters.append(f"val{non_tensor_index}: {type_name(op_desc.dtypes[i])}")
                else:
                    parameters.append(f"val{non_tensor_index}")
                non_tensor_index += 1
        elif op_desc.intents[i] == Intent.OUT:
            if not include_outputs:
                continue
            if op_desc.is_tensor[i]:
                parameters.append(f"out{output_tensor_index}: torch.Tensor")
                output_tensor_index += 1
            else:
                raise ValueError("Unreachable")
        else:
            raise ValueError("Unreachable")
    return ", ".join(parameters)

def parameter_ref_for_wrapper(op_desc: OPDesc, include_outputs=False) -> str:
    """Generate parameter reference for wrapper function"""
    parameters: List[str] = []
    input_tensor_index = 0
    non_tensor_index = 0
    output_tensor_index = 0
    for i in range(op_desc.arity()):
        if op_desc.intents[i] == Intent.IN:
            if op_desc.is_tensor[i]:
                parameters.append(f"in{input_tensor_index}")
                input_tensor_index += 1
            else:
                parameters.append(f"val{non_tensor_index}")
                non_tensor_index += 1
        elif op_desc.intents[i] == Intent.OUT:
            if not include_outputs:
                continue
            if op_desc.is_tensor[i]:
                parameters.append(f"out{output_tensor_index}")
                output_tensor_index += 1
            else:
                raise ValueError("Unreachable")
        else:
            raise ValueError("Unreachable")
    return ", ".join(parameters)

def output_ref_for_wrapper(op_desc: OPDesc) -> str:
    """Generate output variable refernece for wrapper function.
    """
    parameters: List[str] = []
    output_tensor_index = 0
    for i in range(op_desc.arity()):
        if op_desc.intents[i] == Intent.OUT:
            if op_desc.is_tensor[i]:
                parameters.append(f"out{output_tensor_index}")
                output_tensor_index += 1
    return ", ".join(parameters)

def docstring_for_wrapper(op_desc: OPDesc):
    doc = f'"""Generated destination passing style wrapper function with {op_desc.num_input_tensors()} input tensor(s), {op_desc.num_output_tensors()} output tensor(s), {op_desc.num_non_tensor_args()} non tesnor input(s)."""'
    return doc


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import math")
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("from triton import language as tl")
    code.newline()
    code.writeline(
        "from flag_gems.utils.shape_utils import broadcast_shapes, broadcasted_stride, c_contiguous_stride, volume, Stride"
    )
    code.writeline("from flag_gems.utils.libentry import libentry")
    code.newline()
    return code


def generate_functional_pointwise_wrapper(op_desc: OPDesc, wrapper_name: str, destination_passing_func_name: str, code: IndentedBuffer):
    # wrapper signature
    parameters: str = parameter_for_wrapper(op_desc, include_outputs=False)
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        # docstring
        wrapper_docstring = docstring_for_wrapper(op_desc)
        code.writeline(wrapper_docstring)

        shapes_str = ", ".join(f"in{i}.shape" for i in range(op_desc.num_input_tensors()))
        code.writeline(f"shape = broadcast_shapes([{shapes_str}])")

        # output allocation
        num_output_tensor_index = 0
        for i in range(op_desc.arity()):
            if op_desc.intents[i] == Intent.OUT:
                if op_desc.dtypes[i] is None:
                    code.writeline(f"out{num_output_tensor_index} = torch.empty(shape, dtype=in0.dtype, device=in0.device)")
                else:
                    code.writeline(f"out{num_output_tensor_index} = torch.empty(shape, dtype={type_name(op_desc.dtypes[i])}, device=in0.device)")
                num_output_tensor_index += 1

        # call destination_passing_func
        call_str = f"{output_ref_for_wrapper(op_desc)} = {destination_passing_func_name}({parameter_ref_for_wrapper(op_desc, include_outputs=True)})"
        code.writeline(call_str)

        return_str = f"return {output_ref_for_wrapper(op_desc)}"
        code.writeline(return_str)
        code.newline()
    return code


def generate_destination_passing_pointwise_wrapper(op_desc: OPDesc, rank, wrapper_name: str, kernel_name: str, code: IndentedBuffer):
    # wrapper signature
    parameters: str = parameter_for_wrapper(op_desc, include_outputs=True)
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    # task partitioning, 1d task indexing
    tile_size = 512
    num_warps = 4
    if rank == 0:
        tile_size = 32
        num_warps = 1

    with code.indent():
        # docstring
        wrapper_docstring = docstring_for_wrapper(op_desc)
        code.writeline(wrapper_docstring)

        shapes_str = ", ".join(f"in{i}.shape" for i in range(op_desc.num_input_tensors()))
        code.writeline(f"shape = broadcast_shapes([{shapes_str}])")
        code.writeline(f"num_tasks = volume(shape)")
        code.newline()

        # input strides for each input tensor w.r.t. the task index space
        if rank > 0:
            code.writeline("# strides of each tensor argument w.r.t the task space")
            for i in range(op_desc.num_input_tensors()):
                code.writeline(f"in{i}_strides = broadcasted_stride(in{i}.shape, in{i}.stride(), shape)")

            for i in range(op_desc.num_output_tensors()):
                code.writeline(f"out{i}_strides = out{i}.stride()")

            code.newline()

        # grid
        code.writeline("# kernel launch")
        grid_stmt: str = f"grid = triton.cdiv(num_tasks, {tile_size}), 1, 1"
        code.writeline(grid_stmt)

        # launch kernel
        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)


        with code.indent():
            code.writeline(f"{parameter_ref_for_wrapper(op_desc, include_outputs=True)},")

            if rank > 0:
                for i in range(op_desc.num_input_tensors()):
                    s = ", ".join(f"in{i}_strides[{j}]" for j in range(rank))
                    code.writeline(f"{s}, # stride for in{i}")

                for i in range(op_desc.num_output_tensors()):
                    s = ", ".join(f"out{i}_strides[{j}]" for j in range(rank))
                    code.writeline(f"{s}, # stride for out{i}")

                shape_args: str = ", ".join(f"shape[{i}]" for i in range(rank))
                if rank > 0:
                    code.writeline(f"{shape_args}, # task indexing space")
                code.writeline(f"num_tasks, # num tasks")

            code.writeline(f"tile_size={tile_size},")
            code.writeline(f"num_warps={num_warps},")
        code.writeline(")")

        # return
        code.writeline(f"return {output_ref_for_wrapper(op_desc)}")
        code.newline()
    return code


def generate_pointwise_kernel(op_desc: OPDesc, rank, kernel_name: str, code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")

    scalar_fn = op_desc.scalar_fn
    function_ns = NameSpace()
    # signature
    with code.indent():
        input_tensor_index = 0
        non_tensor_index = 0
        output_tensor_index = 0
        for i in range(op_desc.arity()):
            if op_desc.intents[i] == Intent.IN:
                if op_desc.is_tensor[i]:
                    code.writeline(f"in{input_tensor_index}_ptr: tl.pointer_type,")
                    function_ns.create_name(f"in{input_tensor_index}_ptr")
                    input_tensor_index += 1
                else:
                    code.writeline(f"val{non_tensor_index}: {type_name(op_desc.dtypes[i])},")
                    function_ns.create_name(f"val{non_tensor_index}")
                    non_tensor_index += 1
            elif op_desc.intents[i] == Intent.OUT:
                if op_desc.is_tensor[i]:
                    code.writeline(f"out{output_tensor_index}_ptr: tl.pointer_type,")
                    function_ns.create_name(f"out{output_tensor_index}_ptr")
                    output_tensor_index += 1
                else:
                    raise ValueError("Unreachable")
            else:
                raise ValueError("Unreachable")

        if rank > 0:
            for i in range(op_desc.num_input_tensors()):
                for j in range(rank):
                    function_ns.create_name(f"in{i}_stride{j}")
                stride_args = ", ".join(f"in{i}_stride{j}: int" for j in range(rank))
                code.writeline(f"{stride_args}, # strides for in{i}")

            for i in range(op_desc.num_output_tensors()):
                for j in range(rank):
                    function_ns.create_name(f"out{i}_stride{j}")
                stride_args = ", ".join(f"out{i}_stride{j}: int" for j in range(rank))
                code.writeline(f"{stride_args}, # strides for out{i}")

            task_space_args = ", ".join(f"s{i}: int" for i in range(rank))
            for i in range(rank):
                function_ns.create_name(f"s{i}")
            code.writeline(f"{task_space_args}, # task_space")
            code.writeline(f"num_tasks: int,")
            function_ns.create_name("num_tasks")

        code.writeline("tile_size: tl.constexpr,")
        function_ns.create_name("tile_size")
    code.writeline("):")

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
        # volume_expr: str = " * ".join(f"s{i}" for i in range(rank))
        # num_task_stmt: str = f"num_tasks = {volume_expr}"
        # code.writeline(num_task_stmt)
        # function_ns.create_name("num_tasks")

        if rank > 0:
            mask_stmt: str = "mask = tid < num_tasks"
            code.writeline(mask_stmt)
            function_ns.create_name("mask")
            code.newline()

        # reconstruct multi index
        if rank > 0:
            code.writeline("# multi index recontruction")
            for i in reversed(range(rank)):
                code.writeline(f"i{i} = tid % s{i}")
                function_ns.create_name(f"{i}")
                if i > 0:
                    code.writeline(f"tid //= s{i}")
            code.newline()

        # loads
        code.writeline("# loads")
        for i in range(op_desc.num_input_tensors()):
            if rank > 0:
                ptrs_expr: str = " + ".join(f"i{j} * in{i}_stride{j}" for j in range(rank))
                ptrs_expr: str = f"in{i}_ptr + {ptrs_expr}"
                load_stmt: str = f"in{i} = tl.load({ptrs_expr}, mask=mask)"
            else:
                ptrs_expr: str = f"in{i}_ptr"
                load_stmt: str = f"in{i} = tl.load({ptrs_expr})"
            function_ns.create_name(f"in{i}")  # add to the namespace
            code.writeline(load_stmt)
        code.newline()

        # compute
        code.writeline("# compute")

        inputs_to_scalar_fn = []
        outputs_to_scalar_fn = []

        input_tensor_index = 0
        non_tensor_index = 0
        output_tensor_index = 0
        for i in range(op_desc.arity()):
            if op_desc.intents[i] == Intent.IN:
                if op_desc.is_tensor[i]:
                    inputs_to_scalar_fn.append(f"in{input_tensor_index}")
                    input_tensor_index += 1
                else:
                    inputs_to_scalar_fn.append(f"val{non_tensor_index}")
                    non_tensor_index += 1
            elif op_desc.intents[i] == Intent.OUT:
                if op_desc.is_tensor[i]:
                    outputs_to_scalar_fn.append(f"out{output_tensor_index}")
                    output_tensor_index += 1
                else:
                    raise ValueError("Unreachable")
            else:
                raise ValueError("Unreachable")

        compute_body = inline_function(
            scalar_fn,
            inputs_to_scalar_fn,
            outputs_to_scalar_fn,
            function_ns,
        )
        for line in compute_body.strip().splitlines():
            code.writeline(line)
        code.newline()

        # stores
        code.writeline("# stores")
        for i in range(op_desc.num_output_tensors()):
            if rank > 0:
                ptrs_expr: str = " + ".join(f"i{j} * out{i}_stride{j}" for j in range(rank))
                ptrs_expr: str = f"out{i}_ptr + {ptrs_expr}"
                load_stmt: str = f"tl.store({ptrs_expr}, out{i}, mask=mask)"
            else:
                ptrs_expr: str = f"out{i}_ptr"
                load_stmt: str = f"tl.store({ptrs_expr}, out{i})"
            code.writeline(load_stmt)
        code.newline()
    return code

def generate_code(op_desc: OPDesc, inputs, wrapper_name, destination_passing_func_name, kernel_name, code):
    input_tensor_ids = [i for i in range(op_desc.arity()) if (op_desc.is_tensor[i] is True and op_desc.intents[i] == Intent.IN)]
    # print(input_tensor_ids)
    tensor_shapes = [inputs[i].shape for i in input_tensor_ids]
    shape = broadcast_shapes(tensor_shapes)
    rank = len(shape)
    code = generate_imports(code)
    code = generate_functional_pointwise_wrapper(op_desc, wrapper_name, destination_passing_func_name, code)
    code = generate_destination_passing_pointwise_wrapper(op_desc, rank, destination_passing_func_name, kernel_name, code)
    code = generate_pointwise_kernel(op_desc, rank, kernel_name, code)
    return code

class PointwiseDynamicFunction:
    """Utility to generate function for general pointwise operation. It generate wrapper & JITFunction
    which are specialized according to the rank of the task space(the broadcasted shape of all input tensors).
    The generated code are written out to the cache directory (defaults to ~/.flaggems).
    """

    def __init__(self, op_desc: OPDesc):
        self.op_desc = op_desc
        self.scalar_fn_cache_key = op_desc.scalar_fn.cache_key
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args):
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            # generate file & import it
            op_desc = self.op_desc
            code = IndentedBuffer()
            code = generate_code(op_desc, args, "_wrapper", "_wrapper_out", "_jit_function", code)

            file_name = f"pointwise_dynamic_{self.scalar_fn_cache_key}_rank_{key}.py"
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
        return overload(*args)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


def pointwise_dynamic(
    f=None,
    *,
    is_tensor: Optional[List[bool]]=None,
    dtypes: Optional[List[Optional[type]]]=None,
    output_dtypes: Optional[List[type]]=None
):
    def decorator(fn):
        nonlocal is_tensor
        nonlocal dtypes
        nonlocal output_dtypes
        if is_tensor is None:
            num_inputs = len(fn.arg_names)
            is_tensor = [True] * num_inputs
        else:
            num_inputs = len(is_tensor)

        if dtypes is not None:
            assert len(dtypes) == num_inputs
        else:
            dtypes = [None] * num_inputs

        if output_dtypes is None:
            output_dtypes = [None]
            num_outputs = 1
        else:
            num_outputs = len(output_dtypes)

        op_desc = create_description(
            is_tensor=list(is_tensor) + [True] * num_outputs,
            intents = [Intent.IN] * num_inputs + [Intent.OUT] * num_outputs,
            dtypes = list(dtypes) + list(output_dtypes),
            scalar_fn=fn
        )
        return PointwiseDynamicFunction(op_desc)

    if f is not None:
        assert isinstance(f, JITFunction)
        return decorator(f)
    return decorator



if __name__ == "__main__":
    import triton
    @triton.jit
    def trunc(x, min, max):
        return tl.where(x > max, max, tl.where(x < min, min,  x))


    @triton.jit
    def saxpy(x, alpha, y):
        return x * alpha + y


    op_desc = create_description(
        [True, False, True, True],
        [Intent.IN, Intent.IN, Intent.IN, Intent.OUT],
        [None, float, None, None],
        saxpy,
    )

    x = torch.randn(3, 4, device="cuda")
    y = torch.randn(4, device="cuda")

    from flag_gems.utils.code_utils import IndentedBuffer
    code = IndentedBuffer()
    code = generate_code(op_desc, (x, 2.0, y), "_wrapper", "_wrapper_out", "_jit_function", code)
    # print(code.getvalue())

    @pointwise_dynamic(
        is_tensor=[True, False, True],
        dtypes=[None, float, None])
    @triton.jit
    def saxpy(x, alpha, y):
        return x * alpha + y

    out1 = saxpy(x, 2.0, y)
    out2 = x * 2.0 + y
    print(out1)
    print(out2)
    print()

    @pointwise_dynamic(
        is_tensor=[True, False, True])
    @triton.jit
    def saxpy(x, alpha, y):
        return x * alpha + y

    out1 = saxpy(x, 2.0, y)
    out2 = x * 2.0 + y
    print(out1)
    print(out2)
    print()


    @pointwise_dynamic(
        output_dtypes=[bool])
    @triton.jit
    def ge(x, y):
        return x > y

    out1 = ge(x, y)
    out2 = x > y
    print(out1)
    print(out2)
    print()

    @pointwise_dynamic()
    @triton.jit
    def ordinary(x, y):
        return tl.sin(x) + tl.cos(y)

    out1 = ordinary(x, y)
    out2 = torch.sin(x) + torch.cos(y)
    print(out1)
    print(out2)
    print()

    @pointwise_dynamic
    @triton.jit
    def ordinary2(x, y):
        return tl.sin(x) + tl.cos(y)

    out1 = ordinary2(x, y)
    out2 = torch.sin(x) + torch.cos(y)
    print(out1)
    print(out2)
    print()

    @pointwise_dynamic
    @triton.jit
    def ordinary2(x, y):
        return tl.sin(x) + tl.cos(y)

    x = torch.tensor(1.0, device="cuda")
    y = torch.tensor(2.0, device="cuda")
    out1 = ordinary2(x, y)
    out2 = torch.sin(x) + torch.cos(y)
    print(out1)
    print(out2)
    print()

