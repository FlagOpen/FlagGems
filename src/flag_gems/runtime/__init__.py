from . import backend, commom_utils, register
from .config import Config
from .device import device_ctx

global configer, device
configer = Config()
device = device_ctx()


def get_op_tune_config(op_name):
    global configer
    configer = configer or Config()
    return configer.get_op_tune_config(op_name)


def to_register(*args, **kargs):
    register.to_register(*args, **kargs)


def get_device_fn(api_name):
    return backend.gen_torch_device_fn(api_name)


__all__ = ["commom_utils", "backend", "device", "get_op_tune_config", "to_register"]
