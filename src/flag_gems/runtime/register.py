import torch

from . import backend, commom_utils, error
from .backend.device import DeviceDetector

aten_lib = torch.library.Library("aten", "IMPL")


class Register:
    _instance = None

    def __new__(cls, *args, **kargs):
        if cls._instance is None:
            cls._instance = super(Register, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        config,
        user_unused_ops_list=None,
        lib=None,
        debug=True,
    ):
        if not hasattr(self, "initialized"):
            self.initialized = True
            # lib is a instance of torch.library.Library
            self.device = DeviceDetector()
            self.vendor_list = list(commom_utils.vendors_map)
            self.check_backend()
            self.lib = lib
            # reg_key like 'CUDA', reg_bac_key like AutogradCUDA
            self.reg_key = self.device.name.upper()
            self.reg_bac_key = commom_utils.autograd_str + self.reg_key
            self.debug = debug
            self.forward_ops = []
            self.backward_ops = []
            self.vendor_extend_configs = self.get_vendor_extend_op()
            self.vendor_unused_ops_list = self.get_vendor_unused_op()
            self.unused_ops = user_unused_ops_list + self.vendor_unused_ops_list
            self.config = config + self.vendor_extend_configs
            self.config_filter()
            self.for_each()
            if debug:
                self._set_info()

    def config_filter(self):
        self.config = [
            item for item in self.config if item[1].__name__ not in self.unused_ops
        ]

    def check_backend(self):
        is_support = self.device.vendor_name in self.vendor_list
        if is_support is False:
            error.backend_not_support(self.device.name, self.backend_list)

    def get_vendor_extend_op(self):
        if self.device.vendor != commom_utils.vendors.NVIDIA:
            return backend.get_curent_device_extend_op(self.device.vendor_name)
        return ()

    def get_vendor_unused_op(self):
        if self.device.vendor != commom_utils.vendors.NVIDIA:
            return backend.get_curent_device_unused_op(self.device.vendor_name)
        return []

    def register_impl(self, key, fn, has_backward):
        if self.device.vendor != commom_utils.vendors.NVIDIA:
            if key in self.vendor_extend_configs:
                single_item = self.vendor_extend_configs[key]
                _, fn, has_backward = single_item
        device_key = (
            self.reg_bac_key
            if has_backward is commom_utils.Autograd.enable
            else self.reg_key
        )
        self.lib.impl(key, fn, device_key)

    def for_each(self):
        try:
            for key, func, has_backward in self.config:
                if key not in self.unused_ops:
                    self.register_impl(key, func, has_backward)
        except Exception as e:
            error.register_error(e)

    def _set_info(self):
        for _, fn, hasbackward in self.config:
            fn_name = fn.__name__
            self.backward_ops.append(
                fn_name
            ) if hasbackward is commom_utils.Autograd.enable else self.forward_ops.append(
                fn_name
            )

    def get_all_ops(self):
        return self.forward_ops + self.backward_ops if self.debug else []

    def get_forward_ops(self):
        return self.forward_ops if self.debug else []

    def get_backward_ops(self):
        return self.backward_ops if self.debug else []

    def get_unused_ops(self):
        return self.unused_ops

    def get_vendor_name(self):
        return self.device.vendor_name

    def get_current_device(self):
        return self.device.name

    def support_backward(self, fn):
        return fn.__name__ in self.backward_ops
