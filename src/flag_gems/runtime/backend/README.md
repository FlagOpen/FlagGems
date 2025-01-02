## Multiple backend adaptations
### Introduction
The `flag_gems` operator library provides the ability to access multiple backends. If you are a chip vendor and want to merge `your flag_gems code` into our official main branch, you only need to follow these steps to complete `your flag_gems code` merge into our official main branch.

#### step 1:
Create a folder with your vendor's name in the `FlagGems/src/flag_gems/runtime/backend directory`like `_vendorname` and you can You can refer to `FlagGems/src/flag_gems/runtime/backend/_nvidia`.

#### step 2:
Create  Files including but not limited to `__init__.py`, `heuristics_config_utils.py`, `tune_configs.yaml` and folder `ops`

##### step 2.1  `__init__.py`

You can copy `FlagGems/src/flag_gems/runtime/backend/_nvidia/__init__.py` and the  ***only change***  you need to make is to config `VendorInfoBase`
```
VendorInfoBase(
    vendor_name="xxx", device_name="xxx", device_query_cmd="xxx"
)
```
- `vendor_name` is your vendorname like `nvidia`

- `device_name` is your devicename like `cuda`

- `device_query_cmd` is A command that can only be successfully executed on your vendor's device, like `nvidia-smi`

##### step 2.2  `heuristics_config_utils.py`

You should config `triton.heuristics` params in `FlagGems/src/flag_gems/runtime/backend/_vendorname/heuristics_config_utils.py`and you can  refer to `FlagGems/src/flag_gems/runtime/backend/_nvidia/heuristics_config_utils.py`

##### step 2.3  `tune_configs.yaml`
You should config `triton.autotune` params in `FlagGems/src/flag_gems/runtime/backend/_vendorname/tune_configs.yaml` and you can refer to `FlagGems/src/flag_gems/runtime/backend/_nvidia/tune_configs.yaml`

##### step 2.4  `ops`
The directory `ops` stores vendor-customized operator implementations, for example, if you want to create a `custom add op`, you need to put the  implementation in `ops/add.py` and then you should config ops/__init__.py like
```python
from backend_utils import Autograd

from . import add, gelu


def get_specific_ops():
    return (
        ("add.Tensor", add.add, Autograd.disable),
        ("gelu", gelu.gelu, Autograd.enable),
    )


def get_unused_ops():
    return ("cumsum", "cos")
```
and `get_specific_ops` is to get your vendor-customized operators, get_unused_ops() is to get A list of ops that vendors don't want users to use.
