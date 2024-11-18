import triton

from .. import backend, device


class Config:
    def __init__(self):
        self.config = self.get_vendor_tune_config()
        self.gen_key = "gen"
        self.loaded_config = {}
        self.triton_config_default = {"num_stages": 2, "num_warps": 4, "num_ctas": 1}
        self.load_all()

    def load_all(self):
        for key in self.config:
            self.loaded_config[key] = self.get_op_tune_config(key)

    def get_vendor_tune_config(self):
        return backend.get_tune_config(device.device_instance.vendor_name)

    def _gen_impl(self, gen_config, param_config, iteration_keys, current_config, current_iteration):
        if current_iteration == len(iteration_keys):
            return [triton.Config(
                current_config["META"],
                num_warps=current_config["num_warps"],
                num_stages=current_config["num_stages"],
                num_ctas=current_config["num_ctas"],
            )]
        current_config_list = []
        param_key = iteration_keys[current_iteration]
        if param_key in param_config["META"]:
            config_var_key = param_config["META"][param_key]
        else:
            config_var_key = param_config[param_key]
        if isinstance(config_var_key, int):
            key_config = [config_var_key]
        else:
            key_config = gen_config[config_var_key]
        for single_value in key_config:
            if param_key in param_config["META"]:
                current_config["META"][param_key] = single_value
            else:
                current_config[param_key] = single_value
            config_item = self._gen_impl(gen_config, param_config, iteration_keys, current_config, current_iteration + 1)
            current_config_list += config_item
        return current_config_list


    def to_gen_config(self, gen_config):
        param_config = gen_config["param_map"]
        meta_config = param_config["META"]
        iteration_keys = (list(meta_config) + list(param_config))
        iteration_keys.remove('META')
        current_config = {"META":{}}
        current_config.update(self.triton_config_default)
        iteration_count_init = 0
        return self._gen_impl(gen_config, param_config, iteration_keys, current_config, iteration_count_init)

    def get_op_tune_config(self, op_name):
        if op_name in self.loaded_config:
            return self.loaded_config[op_name]
        
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
                    single_config["META"],
                    num_warps=current_config["num_warps"],
                    num_stages=current_config["num_stages"],
                    num_ctas=current_config["num_ctas"],
                )
            )
        return configs
    