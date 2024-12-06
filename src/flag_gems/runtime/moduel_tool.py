import triton

from . import backend
from .backend.device import DeviceDetector

device = DeviceDetector()
tl_extra_module = None
if tl_extra_module is None:
    try:
        tl_extra_module = backend.gen_tl_extra_backend_module()
    except ImportError:
        try:
            tl_extra_module = triton.language.math
        except ImportError:
            tl_extra_module = triton.language.libdevice
