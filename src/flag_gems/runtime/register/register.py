from typing import Optional
import torch
from .. import backend, device, error

aten_lib = torch.library.Library("aten", "IMPL")

class Register:
    def __init__(
        self,
        config: Optional[tuple[tuple]],
        unused_ops_list: Optional[list[str]] = [],
        lib: Optional[any] = None,
        debug: Optional[bool] = False,
    ):
        self.device = device.device_instance
        self.lib = lib
        self.reg_key = self.device.device_name.upper()
        self.reg_bac_key = backend.AUTOGRAD + self.reg_key
        self.vendor_list = list(backend.vendors_map.keys())
        self.vendor_unused_ops = []
        self.vendor_extend_configs = {}
        self.DEBUG = debug
        self.forward_ops = []
        self.backend_ops = []
        self.config = config
        self.vendor_extend_configs = self.get_vendor_extend_op()
        self.vendor_unused_ops = self.get_vendor_unused_op()
        self.unused_ops = unused_ops_list
        # self._check_backend()
        self.for_each(config)
        if debug:
            self._set_info(config)
            self._set_info(self.vendor_extend_configs.values())


    def _check_backend(self):
        is_support = self.device.vendor_name in self.vendor_list
        if is_support is False:
            error.ErrorHandler().backend_not_support(
                self.device.device_name, self.backend_list
            )

    def get_vendor_extend_op(self):
        if self.device.vendor != backend.vendors.NVIDIA:
            return backend.get_curent_device_extend_op(
                self.device.vendor_name
            )
        return {}

    def get_vendor_unused_op(self):
        if self.device.vendor != backend.vendors.NVIDIA:
            return backend.get_curent_device_unused_op(
                self.device.vendor_name
            )
        return {}

    def __pass_register_cond(self, key):
        return key not in self.unused_ops and key not in self.vendor_unused_ops

    def registerImpl(self, key, fn, has_backward):
        if self.device.vendor != backend.vendors.NVIDIA:
            if key in self.vendor_extend_configs:
                single_item = self.vendor_extend_configs[key]
                _, fn, has_backward = single_item
        device_key = (self.reg_bac_key if has_backward is backend.Autograd.enable else self.reg_key)
        self.lib.impl(key, fn, device_key)
        
    def close(self):
        self.lib

    def for_each(self, config):
        try:
            for key, func, has_backward in config:
                # if self.__pass_register_cond(key):
                    self.registerImpl(key, func, has_backward)

        except Exception as e:
            error.PASS(e)
            error.ErrorHandler().register_error()

    def _set_info(self, config):
        for _, fn, hasbackward in config:
            fn_name = fn.__name__
            self.backend_ops.append(
                fn_name
            ) if hasbackward else self.forward_ops.append(fn_name)

    def get_forward_ops(self) -> list[str]:
        return self.forward_ops if self.DEBUG else []

    def get_backend_ops(self) -> list[str]:
        return self.backend_ops if self.DEBUG else []

    def get_unused_ops(self) -> list[str]:
        return self.unused_ops

    def get_vendor_name(self) -> str:
        return self.device.vendor_name

    def get_current_device(self) -> str:
        return self.device.device_name

    def support_backend(self, fn) -> bool:
        return fn.__name__ in self.backend_ops