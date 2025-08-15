from torch_musa import current_device, get_device_capability

from .attention import (
    flash_attention_forward,
    flash_attn_varlen_func,
    scaled_dot_product_attention,
)
from .dropout import dropout, dropout_backward
from .rand import rand
from .rand_like import rand_like

__all__ = [
    "flash_attn_varlen_func",
    "scaled_dot_product_attention",
    "flash_attention_forward",
    "rand",
    "rand_like",
    "dropout",
    "dropout_backward",
]

if get_device_capability(current_device())[0] >= 3:
    from .addmm import addmm
    from .bmm import bmm
    from .gelu import gelu
    from .mm import mm
    from .tanh import tanh

    __all__ += ["gelu"]
    __all__ += ["tanh"]
    __all__ += ["mm"]
    __all__ += ["addmm"]
    __all__ += ["bmm"]
