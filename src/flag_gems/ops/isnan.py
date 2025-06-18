import triton
import triton.language as tl

from ..utils import pointwise_dynamic, tl_extra_shim

try:
    import torch_npu  # noqa: F401

    _isnan = tl.extra.ascend.libdevice.isnan
except:  # noqa: E722
    _isnan = tl_extra_shim.isnan


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isnan_func(x):
    return _isnan(x.to(tl.float32))


def isnan(A):
    print("GEMS ISNAN")
    return isnan_func(A)
