import os
import subprocess

import torch  # noqa: F401

from .. import backend, error
from ..commom_utils import vendors_map


# A singleton class to manage device context.
class DeviceDetector(object):
    _instance = None

    def __new__(cls, *args, **kargs):
        if cls._instance is None:
            cls._instance = super(DeviceDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self, vendor_name=None):
        if not hasattr(self, "initialized"):
            self.initialized = True
            # A list of all available vendor names.
            self.vendor_list = vendors_map.keys()

            # A dataclass instance, get the vendor information based on the provided or default vendor name.
            self.info = self.get_vendor(vendor_name)

            # vendor_name is like 'nvidia', device_name is like 'cuda'.
            self.vendor_name = self.info.vendor_name
            self.name = self.info.device_name
            self.vendor = vendors_map[self.vendor_name]
            self.device_count = backend.gen_torch_device_object(
                self.vendor_name
            ).device_count()

    def get_vendor(self, vendor_name=None) -> tuple:
        # Try to get the vendor name from a quick special command like 'torch.mlu'.
        vendor_name = self._get_vendor_from_quick_cmd()
        if vendor_name is not None:
            return backend.get_vendor_info(vendor_name)
        # Check whether the vendor name is set in the environment variable.
        vendor_from_env = self._get_vendor_from_env()
        if vendor_from_env is not None:
            return backend.get_vendor_info(vendor_from_env)
        try:
            # Obtaining a vendor_info from the methods provided by torch or triton, but is not currently implemented.
            return self._get_vendor_from_lib()
        except Exception:
            return self._get_vendor_from_sys()

    def _get_vendor_from_quick_cmd(self):
        cmd = {
            "cambricon": "mlu",
            "mthreads": "musa",
        }
        for vendor_name, flag in cmd.items():
            if hasattr(torch, flag):
                return vendor_name
        return None

    def _get_vendor_from_env(self):
        device_from_evn = os.environ.get("GEMS_VENDOR")
        return None if device_from_evn not in self.vendor_list else device_from_evn

    def _get_vendor_from_sys(self):
        vendor_infos = backend.get_vendor_infos()
        for single_info in vendor_infos:
            # Get the vendor information by running system commands.
            result = subprocess.run(
                [single_info.device_query_cmd], capture_output=True, text=True
            )
            if result.returncode == 0:
                return single_info
        error.device_not_found()

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
