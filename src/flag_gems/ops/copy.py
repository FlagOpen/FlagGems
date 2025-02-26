import triton

from ..utils import pointwise_dynamic
from ..utils.codegen_config_utils import CodeGenConfig

config_ = CodeGenConfig(
    max_tile_size=1024,
    max_grid_size=(65536, 65536, 65536),
    max_num_warps_per_cta=32,
    prefer_block_pointer=False,
    prefer_1d_tile=False,
    is_scatter_slice = True
    )

@pointwise_dynamic(is_tensor=(True,), promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy(src):
    return src

@pointwise_dynamic(is_tensor=(True,), promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def copy_slice(src):
    return src
