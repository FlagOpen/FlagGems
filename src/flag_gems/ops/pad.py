# import torch
# import triton
# import triton.language as tl


# @triton.jit
# def triton_pad(
#     x_ptr,
#     out_ptr,
#     in_dim0,
#     in_dim1,
#     in_strides0,
#     in_strides1,
#     out_strides0,
#     out_strides1,
#     valid_dim0_start,
#     valid_dim0_end,
#     valid_dim1_start,
#     valid_dim1_end,
#     in_elem_cnt: tl.constexpr,
#     out_elem_cnt: tl.constexpr,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     pid = tl.program_id(0)

#     block_offset = pid * BLOCK_SIZE

#     offset = block_offset + tl.arange(0, BLOCK_SIZE)

#     remaining = offset
#     idx = remaining // out_strides0
#     dst_index_0 = idx
#     remaining = remaining - idx * out_strides0

#     idx = remaining // out_strides1
#     dst_index_1 = idx

#     if_pad_false_mask = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
#     if_pad_true_mask = tl.full((BLOCK_SIZE,), 1, dtype=tl.int32)

#     src_index_0 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
#     src_index_1 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

#     cond = dst_index_0 >= valid_dim0_start and dst_index_0 < valid_dim0_end
#     cond &= dst_index_1 >= valid_dim1_start and dst_index_1 < valid_dim1_end
#     if_pad = tl.where(cond, if_pad_false_mask, if_pad_true_mask).to(tl.int1)

#     src_index_0 = dst_index_0 - valid_dim0_start
#     src_index_1 = dst_index_1 - valid_dim1_start

#     src_index_0 = tl.where(src_index_0 < 0, 0, src_index_0)
#     src_index_1 = tl.where(src_index_1 < 0, 0, src_index_1)

#     src_offset = src_index_0 * in_strides0 + src_index_1 * in_strides1
#     load_cond = src_offset < in_elem_cnt
#     x_val = tl.load(x_ptr + src_offset, mask=(not if_pad) and load_cond, other=0.0)
#     tl.store(out_ptr + offset, x_val, mask=offset < out_elem_cnt)


# def test_triton_pad(x, pad):
#     ndim = x.ndim
#     pad_size = len(pad)
#     assert pad_size % 2 == 0

#     pad_before = [0 for _ in range(ndim)]
#     pad_after = [0 for _ in range(ndim)]

#     pad_pair = pad_size // 2
#     for i in range(pad_pair):
#         pad_before[ndim - i - 1] = pad[2 * i]
#         pad_after[ndim - i - 1] = pad[2 * i + 1]

#     dst_shape = list(x.shape)
#     for i in range(ndim):
#         dst_shape[i] += pad_before[i] + pad_after[i]

#     out = torch.empty(dst_shape, device=x.device, dtype=x.dtype)

#     valid_dim0_start = pad_before[0]
#     valid_dim0_end = dst_shape[0] - pad_before[0]

#     valid_dim1_start = pad_before[1]
#     valid_dim1_end = dst_shape[1] - pad_before[1]

#     BLOCK_SIZE = 256
#     grid = triton.cdiv(out.numel(), BLOCK_SIZE)
#     triton_pad[grid,](
#         x,
#         out,
#         x.shape[0],
#         x.shape[1],
#         x.stride()[0],
#         x.stride()[1],
#         out.stride()[0],
#         out.stride()[1],
#         valid_dim0_start,
#         valid_dim0_end,
#         valid_dim1_start,
#         valid_dim1_end,
#         x.numel(),
#         out.numel(),
#         BLOCK_SIZE,
#         # num_warps=1,
#     )
#     print("triton out is: ", out)


# #     def triton_pad(
# #     x_ptr,
# #     out_ptr,
# #     in_dim0,
# #     in_dim1,
# #     in_strides0,
# #     in_strides1,
# #     out_strides0,
# #     out_strides1,
# #     valid_dim0_start,
# #     valid_dim0_end,
# #     valid_dim1_start,
# #     valid_dim1_end,
# #     in_elem_cnt,
# #     out_elem_cnt,
# #     BLOCK_SIZE: tl.constexpr,
# # )

# # x = torch.ones((4, 4), device="cuda", dtype=torch.float32)
# x = torch.ones((4096, 4096), device="cuda", dtype=torch.float32)

# pad_params = (2, 2)
# # pad_x = torch.nn.functional.pad(x, (2, 2, 2, 2))
# pad_x = torch.nn.functional.pad(x, pad_params)
# print("Pad x is: ", pad_x)

# test_triton_pad(x, pad_params)


import importlib
import os
from typing import Any, Callable, List, Mapping, Optional, Tuple

import torch

from flag_gems.utils.code_cache import cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, NameSpace
from flag_gems.utils.shape_utils import broadcast_shapes

# import triton
# from triton import language as tl


# ------------------ Operation Description ---------------------------
def _type_name(type) -> str:
    "Render typename as string, work for both (bool, int, float, str) and torch.dtype object"
    if type in (bool, int, float, str):
        return type.__name__
    if isinstance(type, torch.dtype):
        return str(type)
    return str(type)


def _check_typed_list(container, type):
    for item in container:
        assert isinstance(item, type)


def _check_sized_list(container, size):
    assert len(container) == size


class OPDesc:
    _num_inputs: int
    _is_tensor: List[bool]
    _dtypes: List[Optional[type]]

    _num_input_tensors: int
    _num_non_tensor_inputs: int

    _num_outputs: int
    _promotion_methods: List[Tuple[int, ...]]

    def __init__(
        self,
        *,
        num_inputs: Optional[int] = None,
        is_tensor: Optional[List[bool]] = None,
        dtypes: Optional[List[Optional[type]]] = None,
        num_outputs: Optional[int] = None,
        promotion_methods: Optional[List[Tuple[int, ...]]] = None,
    ):
        if is_tensor is not None:
            _check_typed_list(is_tensor, bool)
        if dtypes is not None:
            _check_typed_list(dtypes, (type, type(None)))
        # if promotion_methods is None:
        #     raise ValueError(
        #         "No type promotion method provided! You must provide type promotion method for each output!"
        #     )
        # else:
        #     self._promotion_methods = promotion_methods

        if num_inputs is not None:
            self._num_inputs = num_inputs
            if is_tensor is not None:
                _check_sized_list(is_tensor, num_inputs)
                self._is_tensor = is_tensor
            else:
                self._is_tensor = [True] * num_inputs

            if dtypes is not None:
                _check_sized_list(dtypes, num_inputs)
                self._dtypes = dtypes
            else:
                self._dtypes = [None] * num_inputs
        elif is_tensor is not None:
            self._num_inputs = len(is_tensor)
            self._is_tensor = is_tensor
            if dtypes is not None:
                _check_sized_list(dtypes, self._num_inputs)
                self._dtypes = dtypes
            else:
                self._dtypes = [None] * self._num_inputs
        elif dtypes is not None:
            self._num_inputs = len(dtypes)
            self._dtypes = dtypes
            if is_tensor is not None:
                _check_sized_list(is_tensor, self._num_inputs)
                self._is_tensor = is_tensor
            else:
                self._is_tensor = [item is None for item in dtypes]
        else:
            raise ValueError(
                "Cannot make OPDesc when none of (num_inputs, is_tensor, dtypes) is specified."
            )

        # if num_outputs is not None:
        #     self._num_outputs = num_outputs
        #     _check_sized_list(promotion_methods, num_outputs)
        # else:
        #     self._num_outputs = len(promotion_methods)

        self._num_outputs = num_outputs
        assert self._num_inputs >= 1
        assert self._num_outputs >= 1

        self._num_input_tensors = sum(self._is_tensor)
        self._num_non_tensor_inputs = self._num_inputs - self._num_input_tensors

    def num_inputs(self):
        # num of arguments, outputs not included
        return self._num_inputs

    def num_outputs(self):
        return self._num_outputs

    def is_tensor(self, arg_id: int) -> bool:
        return self._is_tensor[arg_id]

    def input_type(self, arg_id) -> Optional[type]:
        return self._dtypes[arg_id]

    def num_input_tensors(self) -> int:
        return self._num_input_tensors

    def num_output_tensors(self) -> int:
        return self._num_outputs

    def num_non_tensor_args(self) -> int:
        return self._num_non_tensor_inputs

    def type_promotion_methods(self) -> List[Tuple[int, ...]]:
        return self._promotion_methods

    # def _match_enum_by_string(
    #     self, input_str: str
    # ) -> utils.ELEMENTWISE_TYPE_PROMOTION_KIND:
    #     for kind in utils.ELEMENTWISE_TYPE_PROMOTION_KIND:
    #         if input_str.lower() == kind.name.lower():
    #             return kind
    #     raise ValueError(f"No matching enum member found for input: {input_str}")

    # def ith_type_promotion_args(self, i) -> List[int]:
    #     return self._promotion_methods[i][:-1]

    # def ith_type_promotion_kind(self, i) -> utils.ELEMENTWISE_TYPE_PROMOTION_KIND:
    #     return self._match_enum_by_string(self._promotion_methods[i][-1])

    def signature(self, outputs_in_arg: bool = False):
        input_types = []
        for is_tensor, dtype in zip(self._is_tensor, self._dtypes):
            if is_tensor:
                input_types.append("Tensor")
            else:
                if dtype is None:
                    input_types.append("scalar")
                else:
                    input_types.append(_type_name(dtype))

        output_types = []
        for _ in range(self.num_outputs()):
            output_types.append("Tensor")
        if outputs_in_arg:
            input_types.extend(output_types)
        sig = f'Pointwise: ({", ".join(input_types)}) -> ({", ".join(output_types)})'
        return sig

    def __str__(self) -> str:
        return self.signature(outputs_in_arg=False)


# --------------------------- pointwise wrapper genration -----------------------------------
def parameter_for_wrapper(op_desc: OPDesc, include_outputs: bool = False) -> str:
    """Generate parameter declaration with type annotation for wrapper function.
    Example: in0: torch.Tensor, val0: float, out0: torch.Tensor
    """
    parameters: List[str] = []

    parameters.append("in0")
    parameters.append("pad")
    return ", ".join(parameters)


def parameter_for_wrapper_out(op_desc: OPDesc, include_outputs: bool = False) -> str:
    """Generate parameter declaration with type annotation for wrapper function.
    Example: in0: torch.Tensor, val0: float, out0: torch.Tensor
    """
    parameters: List[str] = []

    parameters.append("in0")
    parameters.append("out0")
    parameters.append("dst_shape")
    parameters.append("pad_before")
    return ", ".join(parameters)


def ith_parameter_for_type_promotion(op_desc: OPDesc, ith: int) -> str:
    """Generate parameter reference for i-th type promotion rule
    Example: in0, val0, out0
    """
    parameters: List[str] = []

    input_tensor_index = 0
    non_tensor_index = 0
    for i in range(op_desc.num_inputs()):
        if i not in op_desc.ith_type_promotion_args(ith):
            if op_desc._is_tensor[i]:
                input_tensor_index += 1
            else:
                non_tensor_index += 1
            continue
        if op_desc._is_tensor[i]:
            parameters.append(f"in{input_tensor_index}")
            input_tensor_index += 1
        else:
            parameters.append(f"val{non_tensor_index}")
            non_tensor_index += 1

    return ", ".join(parameters)


def parameter_ref_for_wrapper(
    op_desc: OPDesc,
    include_outputs: bool = False,
    include_offset: bool = False,
    include_kwargs: bool = False,
) -> str:
    """Generate parameter reference for wrapper function.
    Example: in0, val0, out0, out0_offset
    """
    parameters: List[str] = []

    # input_tensor_index = 0
    # non_tensor_index = 0
    # for i in range(op_desc.num_inputs()):
    #     if op_desc._is_tensor[i]:
    #         parameters.append(f"in{input_tensor_index}")
    #         input_tensor_index += 1
    #     else:
    #         parameters.append(f"val{non_tensor_index}")
    #         non_tensor_index += 1

    # if include_outputs:
    #     output_tensor_index = 0
    #     for i in range(op_desc.num_outputs()):
    #         parameters.append(f"out{output_tensor_index}")
    #         if include_offset:
    #             parameters.append(f"out{output_tensor_index}_offset")
    #         output_tensor_index += 1

    # if include_kwargs:
    #     parameters.append("**kwargs")

    parameters.append("in0")
    parameters.append("out0")
    parameters.append("dst_shape")
    parameters.append("pad_before")

    return ", ".join(parameters)


def output_ref_for_wrapper(op_desc: OPDesc) -> str:
    """Generate output variable refernece for wrapper function.
    Example: out0, out1
    """
    parameters: List[str] = [f"out{i}" for i in range(op_desc.num_outputs())]
    return ", ".join(parameters)


def docstring_for_functional_wrapper(op_desc: OPDesc):
    doc = f'"""Generated wrapper function with {str(op_desc)}"""'
    return doc


def docstring_for_destination_passing_wrapper(op_desc: OPDesc):
    doc = f'"""Generated wrapper function with {op_desc.signature(outputs_in_arg=True)}"""'
    return doc


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import math")
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("from triton import language as tl")
    code.newline()
    # code.writeline("from flag_gems.utils.shape_utils import (")
    # code.writeline("    broadcast_shapes,")
    # code.writeline("    broadcasted_stride,")
    # code.writeline("    c_contiguous_stride,")
    # code.writeline("    volume,")
    # code.writeline("    Stride,")
    # code.writeline(")")
    code.writeline("from flag_gems.utils.libentry import libentry")
    code.writeline("from flag_gems.utils.type_utils import type_promotion")
    # code.writeline("import torch._prims_common as utils")
    code.newline()
    code.newline()
    return code


def generate_functional_pointwise_wrapper(
    op_desc: OPDesc,
    wrapper_name: str,
    destination_passing_func_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # wrapper signature
    parameters: str = parameter_for_wrapper(op_desc, include_outputs=False)
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    # with code.indent():
    #     # docstring
    #     wrapper_docstring = docstring_for_functional_wrapper(op_desc)
    #     code.writeline(wrapper_docstring)

    #     shapes_str = ", ".join(
    #         f"in{i}.shape" for i in range(op_desc.num_input_tensors())
    #     )
    #     code.writeline(f"shape = broadcast_shapes([{shapes_str}])")

    #     # output allocation
    #     num_output_tensor_index = 0
    #     for i in range(op_desc.num_outputs()):
    #         type_promotion_args = ith_parameter_for_type_promotion(op_desc, i)
    #         k_type_promotion = op_desc.ith_type_promotion_kind(i)
    #         code.writeline(
    #             (
    #                 f"out{num_output_tensor_index} = "
    #                 f"torch.empty(shape, dtype=type_promotion"
    #                 f"({type_promotion_args}, type_promotion=utils.{k_type_promotion})[1], "
    #                 f"device=in0.device)"
    #             )
    #         )
    #         num_output_tensor_index += 1

    #     # call destination_passing_func
    #     output_names: str = output_ref_for_wrapper(op_desc)
    #     call_str = (
    #         f"{output_names} = {destination_passing_func_name}"
    #         f"({parameter_ref_for_wrapper(op_desc, include_outputs=True, include_offset=False, include_kwargs=True)})"
    #     )
    #     code.writeline(call_str)

    #     return_str = f"return {output_names}"
    #     code.writeline(return_str)
    #     code.newline()
    #     code.newline()

    with code.indent():
        # docstring
        wrapper_docstring = docstring_for_functional_wrapper(op_desc)
        code.writeline(wrapper_docstring)

        # ndim = x.ndim
        # pad_size = len(pad)
        # assert pad_size % 2 == 0

        # pad_before = [0 for _ in range(ndim)]
        # pad_after = [0 for _ in range(ndim)]

        # pad_pair = pad_size // 2
        # for i in range(pad_pair):
        #     pad_before[ndim - i - 1] = pad[2 * i]
        #     pad_after[ndim - i - 1] = pad[2 * i + 1]

        # dst_shape = list(x.shape)
        # for i in range(ndim):
        #     dst_shape[i] += pad_before[i] + pad_after[i]

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

        # out = torch.empty(dst_shape, device=x.device, dtype=x.dtype)

        code.writeline(
            ("out0 = torch.empty(dst_shape, device=in0.device, dtype=in0.dtype)")
        )

        # call destination_passing_func
        output_names: str = output_ref_for_wrapper(op_desc)
        call_str = (
            f"{output_names} = {destination_passing_func_name}"
            f"({parameter_ref_for_wrapper(op_desc, include_outputs=True, include_offset=False, include_kwargs=True)})"
        )
        code.writeline(call_str)

        return_str = "return out0"
        code.writeline(return_str)
        code.newline()
        code.newline()

    return code


def generate_destination_passing_pointwise_wrapper(
    op_desc: OPDesc,
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # wrapper signature
    # parameters: str = parameter_for_wrapper(op_desc, include_outputs=True)
    parameters: str = parameter_for_wrapper_out(op_desc, include_outputs=True)

    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        # docstring
        wrapper_docstring = docstring_for_destination_passing_wrapper(op_desc)
        code.writeline(wrapper_docstring)

        code.writeline("BLOCK_SIZE = 256")
        code.writeline("grid = (triton.cdiv(out0.numel(), BLOCK_SIZE), 1, 1)")
        code.newline()

        code.writeline("x_shape = in0.shape")
        code.writeline("in_strides0 = in0.stride()")
        code.writeline("out_strides = out0.stride()")

        # input strides for each input tensor w.r.t. the task index space
        if rank > 0:
            code.writeline("# strides of each tensor argument w.r.t the task space")
            # for i in range(op_desc.num_input_tensors()):
            # code.writeline(
            #     f"in{i}_strides = broadcasted_stride(in{i}.shape, in{i}.stride(), shape)"
            # )

            for i in range(rank):
                code.writeline(f"valid_dim{i}_start = pad_before[{i}]")

                code.writeline(f"valid_dim{i}_end = dst_shape[{i}] - pad_before[{i}]")

            # for i in range(op_desc.num_output_tensors()):
            #     code.writeline(f"if 'out{i}_offset' in kwargs:")
            #     with code.indent():
            #         code.writeline(f"out{i}_offset = kwargs['out{i}_offset']")
            #     code.writeline("else:")
            #     with code.indent():
            #         code.writeline(f"out{i}_offset = 0")

            #     code.writeline(f"if 'out{i}_strides' in kwargs:")
            #     with code.indent():
            #         code.writeline(f"out{i}_strides = kwargs['out{i}_strides']")
            #     code.writeline("else:")
            #     with code.indent():
            #         code.writeline(f"out{i}_strides = out{i}.stride()")
        # else:
        #     for i in range(op_desc.num_output_tensors()):
        #         code.writeline(f"out{i}_offset = 0")

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
                    code.writeline("BLOCK_SIZE, ")
            code.writeline(")")

        code.writeline("return out0")
        code.newline()
        code.newline()
    return code


def generate_pointwise_kernel(
    op_desc: OPDesc,
    # scalar_fn: JITFunction,
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # make the inlined function visible in the context
    # fn_name = scalar_fn.__name__
    # code.writeline(f"from {scalar_fn.__module__} import {fn_name}")
    # code.writeline(f"inlined_f = {fn_name}._scalar_fn")

    code.newline()

    # the decorators
    code.writeline("@libentry()")
    if op_desc.num_non_tensor_args() > 0:
        # we do not specialize non tensor args since they are passed into the inlined function
        # which means that their values may not deserve specialization
        non_specialize_arg_names = [
            f"val{i}" for i in range(op_desc.num_non_tensor_args())
        ]
        code.writeline(f"@triton.jit(do_not_specialize={non_specialize_arg_names})")
    else:
        code.writeline("@triton.jit")

    # signature
    code.writeline(f"def {kernel_name}(")
    function_ns = NameSpace()
    with code.indent():
        input_tensor_index = 0
        non_tensor_index = 0
        output_tensor_index = 0
        # signature: inputs ptrs & non tensor inputs
        for i in range(op_desc.num_inputs()):
            if op_desc.is_tensor(i):
                code.writeline(
                    f"in{input_tensor_index}_ptr: tl.tensor, # of tl.pointer_type"
                )
                function_ns.create_name(f"in{input_tensor_index}_ptr")
                input_tensor_index += 1
            else:
                if op_desc.input_type(i) is not None:
                    code.writeline(
                        f"val{non_tensor_index}: {_type_name(op_desc.input_type(i))},"
                    )
                else:
                    code.writeline(f"val{non_tensor_index},")
                function_ns.create_name(f"val{non_tensor_index}")
                non_tensor_index += 1

        # signature: output ptrs
        for i in range(op_desc.num_outputs()):
            code.writeline(
                f"out{output_tensor_index}_ptr: tl.tensor, # of tl.pointer_type"
            )
            # code.writeline(f"out{output_tensor_index}_offset: int,")
            function_ns.create_name(f"out{output_tensor_index}_ptr")
            # function_ns.create_name(f"out{output_tensor_index}_offset")
            output_tensor_index += 1

        # signature: strides, for each tensor arguments
        # only add this arguments when rank > 0

        # s = ", ".join(f"x_shape[{j}]" for j in range(rank))
        # code.writeline(f"{s}, # shape for x")
        # s = ", ".join(f"in_strides0[{j}]" for j in range(rank))
        # code.writeline(f"{s}, # stride for x")
        # s = ", ".join(f"out_strides[{j}]" for j in range(rank))
        # code.writeline(f"{s}, # stride for out")
        # s = ", ".join(f"valid_dim{j}_start" for j in range(rank))
        # code.writeline(f"{s}, # valid dim start")
        # s = ", ".join(f"valid_dim{j}_end" for j in range(rank))
        # code.writeline(f"{s}, # valid dim end")
        # code.writeline(f"in0.numel(), ")
        # code.writeline(f"out0.numel(), ")

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

            code.writeline("in_elem_cnt: tl.constexpr, # valid dim end")
            code.writeline("out_elem_cnt: tl.constexpr, # valid dim end")
            code.writeline("BLOCK_SIZE: tl.constexpr, # valid dim end")

            # # task space, used to reconstruct multi index
            # task_space_args = ", ".join(f"s{i}: int" for i in range(rank))
            # for i in range(rank):
            #     function_ns.create_name(f"s{i}")
            # code.writeline(f"{task_space_args}, # task_space")

            # # number of tasks, used to compute mask
            # code.writeline("num_tasks: int,")
            # function_ns.create_name("num_tasks")

        # # tile size & tiles_per_cta, gsl style
        # if rank > 0:
        #     code.writeline("tiles_per_cta,")
        #     function_ns.create_name("tiles_per_cta")

        #     code.writeline("tile_size: tl.constexpr,")
        #     function_ns.create_name("tile_size")

        #     code.writeline("one_tile_per_cta: tl.constexpr,")
        #     function_ns.create_name("one_tile_per_cta")
    code.writeline("):")

    # input & output names
    # inputs_to_scalar_fn = []
    # input_tensor_index = 0
    # non_tensor_index = 0
    # for i in range(op_desc.num_inputs()):
    #     if op_desc.is_tensor(i):
    #         inputs_to_scalar_fn.append(f"in{input_tensor_index}")
    #         input_tensor_index += 1
    #     else:
    #         inputs_to_scalar_fn.append(f"val{non_tensor_index}")
    #         non_tensor_index += 1
    # inputs_to_scalar_fn: str = ", ".join(inputs_to_scalar_fn)

    # outputs_to_scalar_fn = [f"out{i}" for i in range(op_desc.num_outputs())]
    # outputs_to_scalar_fn: str = ", ".join(outputs_to_scalar_fn)

    # # function body for rank-0
    # if rank == 0:
    #     with code.indent():
    #         code.writeline("# loads")
    #         for i in range(op_desc.num_input_tensors()):
    #             ptrs_expr: str = f"in{i}_ptr"
    #             load_stmt: str = f"in{i} = tl.load({ptrs_expr})"
    #             function_ns.create_name(f"in{i}")  # add to the namespace
    #             code.writeline(load_stmt)
    #         code.newline()

    #         code.writeline("# compute")
    #         code.writeline(f"{outputs_to_scalar_fn} = inlined_f({inputs_to_scalar_fn})")
    #         code.newline()

    #         code.writeline("# stores")
    #         for i in range(op_desc.num_output_tensors()):
    #             ptrs_expr: str = f"out{i}_ptr + out{i}_offset"
    #             store_stmt: str = f"tl.store({ptrs_expr}, out{i})"
    #             code.writeline(store_stmt)
    #         code.newline()
    #         return code

    # with code.indent():
    #     # get pid
    #     code.writeline("# task id & masking")
    #     pid_stmt = "pid = tl.program_id(0)"
    #     code.writeline(pid_stmt)
    #     function_ns.create_name("pid")

    #     code.writeline("num_ctas = tl.num_programs(0)")
    #     function_ns.create_name("num_ctas")

    #     # get tid (a.k.a task id)
    #     tid_stmt = "init_tid = pid * tile_size + tl.arange(0, tile_size)"
    #     code.writeline(tid_stmt)
    #     function_ns.create_name("init_tid")

    #     # one-tile-per-cta, monolithic kernel style
    #     code.writeline("if one_tile_per_cta: # monolitic kernel style")
    #     with code.indent():
    #         tid_stmt = "tid = init_tid"
    #         code.writeline(tid_stmt)
    #         function_ns.create_name("tid")

    #         # only apply masking when rank > 0
    #         # since we only load a value instead of a block of values when the rank is 0
    #         mask_stmt: str = "mask = tid < num_tasks"
    #         code.writeline(mask_stmt)
    #         function_ns.create_name("mask")
    #         code.newline()

    #         # reconstruct multi index
    #         code.writeline("# multi index recontruction")
    #         for i in reversed(range(rank)):
    #             if i > 0:
    #                 code.writeline(f"i{i} = tid % s{i}")
    #                 code.writeline(f"tid //= s{i}")
    #             else:
    #                 code.writeline(f"i{i} = tid")
    #             function_ns.create_name(f"{i}")
    #         code.newline()

    #         # loads
    #         code.writeline("# loads")
    #         for i in range(op_desc.num_input_tensors()):
    #             ptrs_expr: str = " + ".join(
    #                 f"i{j} * in{i}_stride{j}" for j in range(rank)
    #             )
    #             ptrs_expr: str = f"in{i}_ptr + {ptrs_expr}"
    #             load_stmt: str = f"in{i} = tl.load({ptrs_expr}, mask=mask)"
    #             function_ns.create_name(f"in{i}")  # add to the namespace
    #             code.writeline(load_stmt)
    #         code.newline()

    #         # compute
    #         code.writeline("# compute")
    #         code.writeline(f"{outputs_to_scalar_fn} = inlined_f({inputs_to_scalar_fn})")
    #         code.newline()

    #         # stores
    #         code.writeline("# stores")
    #         for i in range(op_desc.num_output_tensors()):
    #             ptrs_expr: str = " + ".join(
    #                 f"i{j} * out{i}_stride{j}" for j in range(rank)
    #             )
    #             ptrs_expr: str = f"out{i}_ptr + out{i}_offset + {ptrs_expr}"
    #             store_stmt: str = f"tl.store({ptrs_expr}, out{i}, mask=mask)"
    #             code.writeline(store_stmt)

    #     # https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    #     code.writeline("else: # grid-stride-loop style kernel")
    #     with code.indent():
    #         code.writeline("for j in range(0, tiles_per_cta):")
    #         function_ns.create_name("j")
    #         with code.indent():
    #             tid_stmt = "tid = init_tid + j * tile_size * num_ctas"
    #             code.writeline(tid_stmt)
    #             function_ns.create_name("tid")

    #             # only apply masking when rank > 0
    #             # since we only load a value instead of a block of values when the rank is 0
    #             mask_stmt: str = "mask = tid < num_tasks"
    #             code.writeline(mask_stmt)
    #             function_ns.create_name("mask")
    #             code.newline()

    #             # reconstruct multi index
    #             code.writeline("# multi index recontruction")
    #             for i in reversed(range(rank)):
    #                 if i > 0:
    #                     code.writeline(f"i{i} = tid % s{i}")
    #                     code.writeline(f"tid //= s{i}")
    #                 else:
    #                     code.writeline(f"i{i} = tid")
    #                 function_ns.create_name(f"{i}")
    #             code.newline()

    #             # loads
    #             code.writeline("# loads")
    #             for i in range(op_desc.num_input_tensors()):
    #                 ptrs_expr: str = " + ".join(
    #                     f"i{j} * in{i}_stride{j}" for j in range(rank)
    #                 )
    #                 ptrs_expr: str = f"in{i}_ptr + {ptrs_expr}"
    #                 load_stmt: str = f"in{i} = tl.load({ptrs_expr}, mask=mask)"
    #                 function_ns.create_name(f"in{i}")  # add to the namespace
    #                 code.writeline(load_stmt)
    #             code.newline()

    #             # compute
    #             code.writeline("# compute")
    #             code.writeline(
    #                 f"{outputs_to_scalar_fn} = inlined_f({inputs_to_scalar_fn})"
    #             )
    #             code.newline()

    #             # stores
    #             code.writeline("# stores")
    #             for i in range(op_desc.num_output_tensors()):
    #                 ptrs_expr: str = " + ".join(
    #                     f"i{j} * out{i}_stride{j}" for j in range(rank)
    #                 )
    #                 ptrs_expr: str = f"out{i}_ptr + out{i}_offset + {ptrs_expr}"
    #                 store_stmt: str = f"tl.store({ptrs_expr}, out{i}, mask=mask)"
    #                 code.writeline(store_stmt)
    #             code.newline()

    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("block_offset = pid * BLOCK_SIZE")
        code.writeline("offset = block_offset + tl.arange(0, BLOCK_SIZE)")
        code.newline()

        # pid = tl.program_id(0)
        # block_offset = pid * BLOCK_SIZE
        # offset = block_offset + tl.arange(0, BLOCK_SIZE)

        code.writeline("remaining = offset ")
        for i in range(rank):
            code.writeline(f"idx = remaining // out_strides{i}")
            code.writeline(f"dst_index_{i} = idx")
            code.writeline(f"remaining = remaining - idx * out_strides{i}")
            code.newline()

        # remaining = offset
        # idx = remaining // out_strides0
        # dst_index_0 = idx
        # remaining = remaining - idx * out_strides0

        # idx = remaining // out_strides1
        # dst_index_1 = idx

        code.writeline("if_pad_false_mask = tl.zeros((BLOCK_SIZE, ), dtype=tl.int32)")
        code.writeline("if_pad_true_mask = tl.full((BLOCK_SIZE, ), 1, dtype=tl.int32)")

        for i in range(rank):
            code.writeline(f"src_index_{i} = tl.zeros((BLOCK_SIZE, ), dtype=tl.int32)")

        code.writeline(
            "cond = (dst_index_0 >= valid_dim0_start and dst_index_0 < valid_dim0_end) "
        )

        for i in range(1, rank):
            code.writeline(
                f"cond &= (dst_index_{i} >= valid_dim{i}_start and dst_index_{i} < valid_dim{i}_end)"
            )

        # cond = (dst_index_0 >= valid_dim0_start and dst_index_0 < valid_dim0_end)
        # cond &= (dst_index_1 >= valid_dim1_start and dst_index_1 < valid_dim1_end)

        code.writeline(
            "if_pad = tl.where(cond, if_pad_false_mask, if_pad_true_mask).to(tl.int1)"
        )

        for i in range(rank):
            code.writeline(f"src_index_{i} = dst_index_{i} - valid_dim{i}_start ")

        # src_index_0 = dst_index_0 - valid_dim0_start
        # src_index_1 = dst_index_1 - valid_dim1_start

        for i in range(rank):
            code.writeline(
                f"src_index_{i} = tl.where(src_index_{i} < 0, 0, src_index_{i})"
            )

        # src_index_0 = tl.where(src_index_0 < 0, 0, src_index_0)
        # src_index_1 = tl.where(src_index_1 < 0, 0, src_index_1)

        code.writeline("src_offset = src_index_0 * in_strides0")
        for i in range(1, rank):
            code.writeline(f"src_offset += src_index_{i} * in_strides{i}")

        # src_offset = src_index_0 * in_strides0 + src_index_1 * in_strides1

        code.writeline("load_cond = src_offset < in_elem_cnt")
        code.writeline(
            "x_val = tl.load(in0_ptr + src_offset, mask=(not if_pad) and load_cond, other=0.0)"
        )
        code.writeline("tl.store(out0_ptr + offset, x_val, mask=offset < out_elem_cnt)")

        # load_cond = src_offset < in_elem_cnt
        # x_val = tl.load(x_ptr + src_offset, mask=(not if_pad) and load_cond, other=0.0)
        # tl.store(out_ptr + offset, x_val, mask=offset < out_elem_cnt)

    return code


def generate_code(
    op_desc: OPDesc,
    # scalar_fn: JITFunction,
    inputs: Tuple[Any],
    wrapper_name: str,
    destination_passing_func_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # assert (
    #     len(inputs) == op_desc.num_inputs()
    # ), "the number of inputs does not match {str(op_desc)}"
    input_tensor_ids = [i for i in range(op_desc.num_inputs()) if op_desc.is_tensor(i)]
    tensor_shapes = [inputs[i].shape for i in input_tensor_ids]
    shape = broadcast_shapes(tensor_shapes)
    rank = len(shape)

    # the only runtime determined factor is the rank of the task space
    code = generate_imports(code)
    code = generate_functional_pointwise_wrapper(
        op_desc, wrapper_name, destination_passing_func_name, code
    )
    code = generate_destination_passing_pointwise_wrapper(
        op_desc, rank, destination_passing_func_name, kernel_name, code
    )
    # code = generate_pointwise_kernel(op_desc, scalar_fn, rank, kernel_name, code)

    code = generate_pointwise_kernel(op_desc, rank, kernel_name, code)
    return code


class PadFunction:
    """Utility to generate function for general pointwise operation. It generate wrapper & JITFunction
    which are specialized according to the rank of the task space(the broadcasted shape of all input tensors).
    The generated code are written out to the cache directory (defaults to ~/.flaggems).
    """

    # def __init__(self, op_desc: OPDesc, scalar_fn: JITFunction):
    def __init__(self, op_desc: OPDesc):
        self._op_desc = op_desc

        # assert isinstance(scalar_fn, JITFunction)
        # self._scalar_fn = scalar_fn
        # self._scalar_fn_cache_key = scalar_fn.cache_key
        self.pid = os.getpid()

        # instantiated & cached overloads
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
                self._op_desc,
                # self._scalar_fn,
                args,
                "_wrapper",
                "_wrapper_out",
                "_jit_function",
                code,
            )

            # file_name = f"constant_pad_{self._scalar_fn_cache_key}_rank_{key}_pid_{self.pid}.py"

            file_name = f"constant_pad_rank_{key}_pid_{self.pid}.py"

            with open(cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            # load
            # spec = importlib.util.spec_from_file_location(
            #     f"_gen_module_{self._scalar_fn_cache_key}_rank_{key}_pid_{self.pid}",
            #     f.name,
            # )

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


# def pad_dynamic(
#     f: Optional[JITFunction] = None,
#     *,
#     num_inputs: Optional[int] = None,
#     is_tensor: Optional[List[bool]] = None,
#     dtypes: Optional[List[Optional[type]]] = None,
#     num_outputs: Optional[int] = None,
#     promotion_methods: Optional[Tuple[int, ...]] = None,
# ):
#     def decorator(fn):
#         nonlocal num_inputs
#         if (num_inputs is None) and (is_tensor is None) and (dtypes is None):
#             num_inputs = len(fn.arg_names)
#         op_desc = OPDesc(
#             num_inputs=num_inputs,
#             is_tensor=is_tensor,
#             dtypes=dtypes,
#             num_outputs=num_outputs,
#             promotion_methods=promotion_methods,
#         )
#         # return PadFunction(op_desc, fn)
#         return PadFunction(op_desc)

#     if f is not None:
#         return decorator(f)
#     return decorator

op_desc = OPDesc(
    num_inputs=1,
    is_tensor=None,
    dtypes=None,
    num_outputs=1,
    promotion_methods=None,
)

pad_func = PadFunction(op_desc)


x = torch.ones((4, 4), device="cuda", dtype=torch.float32)
pad_params = (2, 2)

triton_pad_x = pad_func(x, pad_params)
print("triton pad x is: ", triton_pad_x)
