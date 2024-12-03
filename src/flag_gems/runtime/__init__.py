from . import backend, commom_utils, register
from .configloader import ConfigLoader
from .device import DeviceDetector

configer = ConfigLoader()
device = DeviceDetector()


def get_op_tune_config(op_name):
    return configer.get_op_tune_config(op_name)


def to_register(*args, **kargs):
    register.to_register(*args, **kargs)


def get_device_fn(api_name):
    return backend.gen_torch_device_fn(api_name)


__all__ = ["commom_utils", "backend", "device", "get_op_tune_config", "to_register"]
