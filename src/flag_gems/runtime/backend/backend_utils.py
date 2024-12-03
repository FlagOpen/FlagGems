import json
import os
from dataclasses import dataclass


# Metadata template,  Each vendor needs to specialize instances of this template
@dataclass
class vendorInfoBase:
    vendor_name: str
    device_name: str
    cmd: str


def get_tune_config(vendor_name, file_mode="r"):
    vendor_name = "_" + vendor_name
    script_path = os.path.abspath(__file__)
    file_path = os.path.dirname(script_path) + "/" + vendor_name + "/tune_configs.json"
    with open(file_path, file_mode) as file:
        config = json.load(file)
    return config
