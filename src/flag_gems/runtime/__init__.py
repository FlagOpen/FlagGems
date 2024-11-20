from . import backend, commom_utils, device, register, tune_config


def get_op_tune_config(op_name):
    return tune_config.config_instance.get_op_tune_config(op_name)


def to_register(*args, **kargs):
    register.to_register(*args, **kargs)


def get_device_fn(api_name):
    return backend.gen_torch_device_fn(api_name)


def get_device_conut():
    return get_device_fn("device_count")


__all__ = ["commom_utils", "backend", "device", "get_op_tune_config", "to_register"]
