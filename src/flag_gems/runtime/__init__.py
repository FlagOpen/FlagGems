from . import backend, commom_utils, error
from .backend.device import DeviceDetector
from .configloader import ConfigLoader

config_loader = ConfigLoader()
device = DeviceDetector()

"""
The dependency order of the sub-directory is strict, and changing the order arbitrarily may cause errors.
"""

# torch_device_fn is like 'torch.cuda' object
backend.set_torch_backend_device_fn(device.vendor_name)
torch_device_fn = backend.gen_torch_device_object()

# torch_backend_device is like 'torch.backend.cuda' object
torch_backend_device = backend.get_torch_backend_device_fn()


def get_tuned_config(op_name):
    return config_loader.get_tuned_config(op_name)


def get_heuristic_config(op_name):
    return config_loader.heuristics_config[op_name]


def replace_customized_ops(_globals):
    if device.vendor != commom_utils.vendors.NVIDIA:
        customized_op_infos = backend.get_curent_device_extend_op()
        for single_op_info in customized_op_infos:
            op_fun = single_op_info[1]
            op_fun_name = op_fun.__name__
            try:
                _globals[op_fun_name] = op_fun
            except RuntimeError as e:
                error.customized_op_replace_error(e)


__all__ = [
    "commom_utils",
    "backend",
    "device",
    "get_tuned_config",
    "get_heuristic_config",
]
