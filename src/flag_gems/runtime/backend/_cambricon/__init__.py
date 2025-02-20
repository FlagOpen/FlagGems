import torch
import torch_mlu

from flag_gems.runtime.backend.backend_utils import VendorInfoBase  # noqa: E402

from .utils import (
    DEVICE_COUNT,
    MAX_GRID_SIZE_X,
    MAX_GRID_SIZE_Y,
    MAX_GRID_SIZE_Z,
    MAX_GRID_SIZES,
    MAX_NRAM_SIZE,
    TOTAL_CLUSTER_NUM,
    TOTAL_CORE_NUM,
)

try:
    from torch_mlu.utils.model_transfer import transfer
except ImportError:
    pass

vendor_info = VendorInfoBase(
    vendor_name="cambricon", device_name="mlu", device_query_cmd="cnmon"
)

CUSTOMIZED_UNUSED_OPS = (
    "randperm",  # skip now
    "sort",  # skip now
    "multinomial",  # skip now
    "_upsample_bicubic2d_aa",  # skip now
    "batch_norm",  #
    "pad",
    "constant_pad_nd",  #
)

__all__ = ["*"]
