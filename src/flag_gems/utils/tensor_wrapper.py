import math

import torch

import flag_gems


class TypedPtr:
    """This is a minimal requirement for a type to be treated as a tensor in triton jit
    function. Basically it is a typed pointer, without knowning the device, size, shape,
    strides, etc.
    """

    def __init__(self, ptr: int, dtype: torch.dtype):
        self.ptr = ptr
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self.ptr

    def untyped_storage(self):
        return self

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, offset: int = 0):
        return cls(tensor.data_ptr() + tensor.element_size() * offset, tensor.dtype)

    @classmethod
    def reinterpret_tensor(cls, tensor: torch.Tensor, dtype: torch.dtype, offset=0):
        return cls(tensor.data_ptr() + dtype.itemsize * offset, dtype)


class StridedBuffer:
    """A drop-in replacement of torch.Tensor that can be used in wrapper generated by
    PointwiseDynamicFunction. It allows us to use a different shape, stride, data
    pointer that that of the base tensor.

    It is a kind of reinterpretation of the base tensor. We make this class since we
    cannot get a Tensor view with negative strides via torch APIs, while we need this
    to implement flip op.

    Although generated code can accept torch.Tensor & StridedBuffer, but StridedBuffer
    may not have all the methods as torch.Tensors do. We add some attributes & methods
    with the same name as torch.Tensor, which are used in the generated code. But we
    may not cover all the methods, add one if what you need is missing here.

    And can also be used in triton kernels since it also has dtype & data_ptr().
    """

    def __init__(
        self, base: torch.Tensor, shape=None, strides=None, dtype=None, offset=0
    ):
        self._base = base
        self.dtype = dtype or base.dtype

        if offset == 0:
            self._data_ptr = self._base.data_ptr()
        else:
            # TODO[kunlunxin]: we will upgrade torch version in 2025.04
            if flag_gems.vendor_name == "kunlunxin":

                def get_dtype_bytes(dtype):
                    if dtype.is_floating_point:
                        return int(torch.finfo(dtype).bits / 8)
                    else:
                        return int(torch.iinfo(dtype).bits / 8)

                offset = get_dtype_bytes(self.dtype) * offset
            else:
                offset = self.dtype.itemsize * offset

            self._data_ptr = self._base.data_ptr() + offset
        self.shape = tuple(shape if shape is not None else self._base.shape)
        self._strides = tuple(strides if strides is not None else self._base.stride())
        self.device = self._base.device
        self.ndim = len(self.shape)

    def stride(self):
        return self._strides

    def size(self):
        return self.shape

    def element_size(self):
        return self.dtype.itemsize

    def numel(self):
        return math.prod(self.shape)

    def dim(self):
        return self.ndim

    def unwrap(self):
        return self._base

    def data_ptr(self):
        return self._data_ptr

    def untyped_storage(self):
        return self._base.untyped_storage()
