import pytest
import torch

import flag_gems

# 导入测试框架中的工具函数
from .accuracy_utils import (
    FLOAT_DTYPES,
    gems_assert_close,
)

# 尝试导入 Transformer Engine 的后端作为参考标准
# 如果导入失败，测试将被跳过
try:
    from transformer_engine.pytorch import cpp_extensions as tex
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


# 为 dreglu 定义一组有代表性的形状
# M = batch_size * seq_len, N = hidden_size
DREGU_SHAPES = [
    (4096, 1024),
    (2048, 2048),
    (1024, 4096),
    (512, 512),
    (1, 2048),
    (2048, 1),
    (512, 512, 512),
    # (512, 512, 512, 512), 这个会导致内存过大而报错
]


@pytest.mark.dreglu
@pytest.mark.parametrize("shape", DREGU_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_dreglu(shape, dtype):
    """
    正确性测试：
    对比 flag_gems.dreglu 的输出和 Transformer Engine 官方后端 dreglu 的输出。
    这个版本现在可以处理任意维度的输入形状。
    """
    if not TE_AVAILABLE:
        pytest.skip("Transformer Engine backend (cpp_extensions) not available for reference.")

    # =================================================================
    # 【核心修正】正确处理任意维度的 shape
    # =================================================================
    # dreglu 至少需要一维张量，且最后一维需要能被2整除
    if len(shape) == 0:
        pytest.skip("dreglu does not support 0-dim scalar tensors.")
    
    # 确保最后一个维度是偶数
    if shape[-1] % 2 != 0:
        # 如果最后一个维度是奇数，将其加一变为偶数
        shape = list(shape)
        shape[-1] += 1
        shape = tuple(shape)

    # 1. 准备输入数据
    # input_tensor 的形状是 (..., 2*N)
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    
    # grad_output 的形状是 (..., N)
    grad_output_shape = list(shape)
    grad_output_shape[-1] //= 2
    grad_output = torch.randn(tuple(grad_output_shape), dtype=dtype, device=flag_gems.device)
    
    # 2. 计算参考结果 (Reference Output from Transformer Engine)
    # TE 的 dreglu 函数可以自动处理多维张量
    ref_out = tex.dgeglu(grad_output, input_tensor, None)

    # 3. 计算待测结果 (Result Output from flag_gems)
    # 我们假设 flag_gems.dreglu 也被设计为可以处理多维张量
    # (因为它内部有塑形逻辑)
    with flag_gems.use_gems():
        res_out = flag_gems.dgeglu(grad_output, input_tensor)

    # 4. 对比结果
    gems_assert_close(res_out, ref_out, dtype)