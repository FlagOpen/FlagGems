from backend_utils import Autograd

from . import gelu, tanh

from torch_musa import current_device, get_device_capability


def get_specific_ops():
    if get_device_capability(current_device())[0] >= 3:
        return (
            ("gelu", gelu.gelu, Autograd.enable),
            ("tanh", tanh.tanh, Autograd.enable),
        )
    else:
        return ()


def get_unused_ops():
    return ()


__all__ = ["get_specific_ops", "get_unused_ops"]
