from . import backend, commom_utils, register
from .backend.device import DeviceDetector
from .configloader import ConfigLoader

configer = ConfigLoader()
device = DeviceDetector()


def get_triton_config(op_name):
    return configer.get_triton_config(op_name)


def to_register(*args, **kargs):
    register.to_register(*args, **kargs)


def get_device_fn(api_name):
    return backend.gen_torch_device_fn(api_name)


__all__ = ["commom_utils", "backend", "device", "get_triton_config", "to_register"]
