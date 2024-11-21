import ast
import functools
import importlib
import os
import sys

from ..commom_utils import AUTOGRAD, Autograd, vendors, vendors_map

vendors = vendors
AUTOGRAD = AUTOGRAD
Autograd = Autograd
vendor_module_name = None
vendor_module = None
device_name = None
device_fn_cache = {}


def get_codegen_result(code, result_key):
    parsed_ast = ast.parse(code)
    compiled_code = compile(parsed_ast, filename="<ast>", mode="exec")
    try:
        exec(compiled_code, globals())
    except Exception as e:
        RuntimeError(e)
    return globals()[result_key]


@functools.lru_cache(maxsize=32)
def gen_torch_tensor_attr_res(tensor, attr_name):
    global device_name
    device_name = device_name or get_vendor_info()[1]
    code = f"""
import torch
res = {tensor}.{attr_name}
    """
    return get_codegen_result(code, "res")


def gen_torch_device_fn(api_name):
    global device_name
    device_name = device_name or get_vendor_info()[1]
    if api_name in device_fn_cache:
        return device_fn_cache[api_name]
    code = f"""
import torch
fn = torch.{device_name}.{api_name}()
"""
    fn = get_codegen_result(code, "fn")
    device_fn_cache[api_name] = fn
    return fn


def get_vendor_module(vendor_name, query=False):
    def get_module(vendor_name):
        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.dirname(current_file_path)
        sys.path.append(current_dir_path)
        return importlib.import_module(vendor_name)

    if query:
        return get_module(vendor_name)

    global vendor_module_name, vendor_module
    if vendor_module_name is None:
        vendor_module_name = vendor_name
        vendor_module = get_module("_" + vendor_name)


def get_vendor_info(vendor_name=None, query=False):
    if query:
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
        except Exception as e:  # noqa: F841
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
    return vendor_module.Op.get_unused_op()


def get_tune_config(vendor_name=None) -> dict:
    global vendor_module
    get_vendor_module(vendor_name)
    return vendor_module.config.get_tune_config()


__all__ = ["*"]
