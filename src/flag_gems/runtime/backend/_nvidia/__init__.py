import json
import os

from .ops import *  # noqa: F403


class device:
    @staticmethod
    def get_vendor_info():
        return ("nvidia", "cuda", "nvidia-smi")


class Op:
    @staticmethod
    def get_register_op_config():
        return (("add.Tensor", add, False),)

    @staticmethod
    def get_unused_op():
        return ("cumsum", "cos")


class config:
    @staticmethod
    def get_tune_config(file_mode="r"):
        script_path = os.path.abspath(__file__)
        file_path = os.path.dirname(script_path) + "/tune_configs.json"
        with open(file_path, file_mode) as file:
            config = json.load(file)
        return config


__all__ = ["device", "Op", "config"]
