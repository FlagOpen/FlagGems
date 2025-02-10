from .gelu import gelu
from .tanh import tanh

from torch_musa import current_device, get_device_capability


__all__ = []

if get_device_capability(current_device())[0] >= 3:
    __all__.append("gelu")
    __all__.append("tanh")


