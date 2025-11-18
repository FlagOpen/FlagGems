import argparse

from . import backend


class OpDispatcher:
    """Operator Dispatcher for selecting vendor-specific operators and configurations."""

    def __init__(self, attrs=None):
        self.attrs = attrs or {}
        self.is_dispatch = False
        self.is_dispatch_configs = False
        self.is_dispatch_operators = False
        self.is_debug = False
        self.operator_vendor = None
        self.config_vendor = None
        self.configurations = None
        self.operators = None

        self._detect_user_action()
        if self.is_user_dispatch or attrs:
            self.is_dispatch = True
            self.set_init_attr()
            self._load_configs()
            self._load_operators()

    def _detect_user_action(self):
        """Detect whether user provided command-line args."""

        args = self.get_cmd_args()
        vendor = args.vendor
        self.is_debug = args.debug or self.is_debug
        if args.ops or args.configs or vendor:
            self.is_user_dispatch = True
            self.attrs = {
                "operators": vendor or args.ops,
                "configurations": vendor or args.configs,
            }
        else:
            self.is_user_dispatch = False

    def set_init_attr(self, attrs=None):
        attrs = attrs or self.attrs
        self.operator_vendor = attrs["operators"]
        self.config_vendor = attrs["configurations"]

    def _load_configs(self, config_vendor=None):
        """Load configuration sets for vendor."""

        vendor = self.config_vendor or config_vendor
        if vendor:
            self.configurations = {
                "heuristic": backend.get_heuristic_config(vendor_name=vendor),
                "autotune": backend.get_tune_config(vendor_name=vendor),
            }
            self.is_dispatch_configs = True
            print(
                f"\033[92m[INFO]\033[0m : \033[93m configurations of {vendor} vendor has been loaded\033[0m"
            )

    def _load_operators(self, operator_vendor=None):
        """Load operator implementations for vendor."""

        vendor = self.operator_vendor or operator_vendor
        if vendor:
            self.operators = backend.get_current_device_extend_op(vendor_name=vendor)
            self.is_dispatch_operators = True
            print(
                f"\033[92m[INFO]\033[0m :  \033[93mOperators of {vendor} vendor has been loaded\033[0m"
            )

    def get_cmd_args(self):
        parser = argparse.ArgumentParser(description="...")
        parser.add_argument("--ops", type=str, required=False)
        parser.add_argument("--configs", type=str, required=False)
        parser.add_argument("--vendor", type=str, required=False)
        parser.add_argument("--debug", type=str, required=False)
        args = parser.parse_args()
        return args


op_dispatcher = OpDispatcher()
