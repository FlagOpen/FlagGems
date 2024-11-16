from . import commom_utils
from . import backend
from . import device
from . import tune_config
from . import register


def get_op_tune_config(op_name):
    return tune_config.config_instance.get_op_tune_config(op_name)

def to_register(*args, **kargs):
    register.to_register(*args, **kargs)

__all__ = ["commom_utils", "backend", "device", "get_op_tune_config", "to_register"]