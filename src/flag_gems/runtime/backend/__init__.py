import ast
import os
# from .. import error
import sys
import importlib
from ..commom_utils import vendors_map, AUTOGRAD, vendors, Autograd
vendors = vendors
AUTOGRAD = AUTOGRAD
Autograd = Autograd
vendor_module_name = None
vendor_module = None

def get_vendor_module(vendor_name, query=False):
    def get_module(vendor_name):
        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.dirname(current_file_path)
        sys.path.append(current_dir_path)
        return importlib.import_module(vendor_name)

    if query == True:
        return get_module(vendor_name)
    
    global vendor_module_name, vendor_module
    if vendor_module_name is None:
        vendor_module_name = vendor_name
        vendor_module = get_module("_"+vendor_name)

def get_device_guard_fn(vendor_name=None):
    global vendor_module
    get_vendor_module(vendor_name)
    return vendor_module.device.get_torch_device_guard_fn()

def get_vendor_info(vendor_name=None, query=False):
    if query == True:
        return get_vendor_module(vendor_name, query).device.get_vendor_info()
    global vendor_module
    get_vendor_module(vendor_name)
    return vendor_module.device.get_vendor_info()

def get_vendor_infos() -> list:
    infos = []
    for vendor_name in vendors_map:
        vendor_name = "_" + vendor_name
        try:
            single_info = get_vendor_info(vendor_name, query=True)
            infos.append(single_info + (vendors_map[single_info[0]],))
        except Exception as e:
            pass
    return infos

def get_curent_device_extend_op(vendor_name=None) -> dict:
    global vendor_module
    get_vendor_module(vendor_name)
    tuples = vendor_module.Op.get_register_op_config()
    configs = {}
    for item in tuples:
        configs[item[0]] = item
    return configs

def get_curent_device_unused_op(vendor_name=None) -> list:
    global vendor_module
    get_vendor_module(vendor_name)
    return  vendor_module.Op.get_unused_op()


def get_tune_config(vendor_name=None) -> dict:
    global vendor_module
    get_vendor_module(vendor_name)
    return  vendor_module.config.get_tune_config()

def device_guard_fn(vendor_name=None):
    return get_device_guard_fn(vendor_name)


__all__ = ["*"]

