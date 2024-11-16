import triton

from .. import backend, device


class Config:
    def __init__(self):
        self.config = self.get_vendor_tune_config()
        self.gen_key = "gen"
        self.triton_config_default = {"num_stages": 2, "num_warps": 4, "num_ctas": 1}

    def get_vendor_tune_config(self):
        return backend.get_tune_config(device.device_instance.vendor_name)

    def _gen_impl(gen_config, iteration_keys, current_iteration):
        param_key = iteration_keys[current_iteration]
        if param_key in gen_config["META"]:
            key_config = gen_config["META"][param_key]
        for single_value in key_config:
            pass

    def to_gen_config(self, gen_config):
        meta_config = gen_config["META"]
        iteration_keys = (list(meta_config) + list(gen_config)).remove("META")
        iteration_count = len(iteration_keys)
        return self._gen_impl(gen_config, iteration_keys, iteration_count)

    def get_op_tune_config(self, op_name):
        current_op_configs = self.config[op_name]
        configs = []
        if len(current_op_configs) == 0:
            return configs

        if len(current_op_configs) == 1:
            single_config = current_op_configs[0]
            if self.gen_key in single_config:
                return self.to_gen_config(single_config)

        for single_config in current_op_configs:
            current_config = self.triton_config_default
            for default_param in current_config:
                if default_param in single_config:
                    current_config[default_param] = single_config[default_param]

            configs.append(
                triton.Config(
                    single_config["size_config"],
                    num_warps=current_config["num_warps"],
                    num_stages=current_config["num_stages"],
                    num_ctas=current_config["num_ctas"],
                )
            )
        return configs