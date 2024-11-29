import os
import subprocess

import torch  # noqa: F401

from . import backend, error
from .commom_utils import quick_special_cmd, vendors_map

global device_instance
device_instance = None


class device_ctx:
    def __init__(self, vendor_name=None):
        self.vendor_list = vendors_map.keys()
        self.device_info = self.get_vendor(vendor_name)
        self.vendor_name = self.device_info.vendor_name
        self.device_name = self.device_info.device_name
        self.vendor = vendors_map[self.vendor_name]
        self.device_count = backend.gen_torch_device_fn(
            "device_count", self.vendor_name
        )()

    def get_vendor(self, vendor_name=None) -> tuple:
        vendor_name = self._get_vendor_from_quick_cmd()
        if vendor_name is not None:
            return backend.get_vendor_info(vendor_name)
        vendor_from_env = self._get_vendor_from_env()
        if vendor_from_env is not None:
            return backend.get_vendor_info(vendor_from_env)
        try:
            return self._get_vendor_from_lib()
        except Exception:
            return self._get_vendor_from_sys()

    def _get_vendor_from_quick_cmd(self):
        for vendor_name, cmd in quick_special_cmd.items():
            try:
                exec(cmd, globals())
                return vendor_name
            except Exception:
                pass
        return None

    def _get_vendor_from_env(self):
        device_from_evn = os.environ.get("GEMS_VENDOR")
        return None if device_from_evn not in self.vendor_list else device_from_evn

    def _get_vendor_from_sys(self):
        vendor_infos = backend.get_vendor_infos()
        for single_info in vendor_infos:
            result = subprocess.run([single_info.cmd], capture_output=True, text=True)
            if result.returncode == 0:
                return single_info
        error.device_not_found()

    def get_device_name(self):
        return self.device_name

    def get_vendor_name(self):
        return self.vendor_name

    def _get_vendor_from_lib(self):
        # Reserve the associated interface for triton or torch
        # although they are not implemented yet.
        # try:
        #     return triton.get_vendor_info()
        # except Exception:
        #     return torch.get_vendor_info()
        raise RuntimeError("The method is not implemented")


device_instance = device_instance or device_ctx()
