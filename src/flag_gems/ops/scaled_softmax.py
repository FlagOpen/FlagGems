import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

autotune_configs = [
    # 小BLOCK_Q配置 (1-16)
    triton.Config({"BLOCK_Q": 1, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 1, "BLOCK_K": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 1, "BLOCK_K": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 1, "BLOCK_K": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 1, "BLOCK_K": 2048}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 2, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 4, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 4, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 4, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 8, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 8, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 16, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    # 大BLOCK_Q配置 (32-128) - 使用更多warps和stages
    triton.Config({"BLOCK_Q": 32, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_Q": 32, "BLOCK_K": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_Q": 32, "BLOCK_K": 512}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_Q": 64, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_Q": 64, "BLOCK_K": 256}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_Q": 64, "BLOCK_K": 512}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_Q": 64, "BLOCK_K": 1024}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_Q": 128, "BLOCK_K": 512}, num_warps=8, num_stages=4),
]


@libentry()
@triton.autotune(configs=autotune_configs, key=["query_seq_len", "key_seq_len"])
@triton.jit
def scaled_softmax_forward_kernel(
    output_ptr,
    input_ptr,
    scale_factor,
    query_seq_len,
    key_seq_len,
    stride_b,
    stride_h,
    stride_q,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Scaled Softmax Forward Kernel
    实现: output = softmax(input * scale_factor)

    Args:
        output_ptr: 输出张量指针
        input_ptr: 输入张量指针
        scale_factor: 缩放因子
        query_seq_len: 查询序列长度
        key_seq_len: 键序列长度
        stride_b: batch维度步长
        stride_h: head维度步长
        stride_q: query维度步长
        BLOCK_Q: Q维度分块大小
        BLOCK_K: K维度分块大小
    """

    # --- 1. 获取程序ID ---
    query_seq_tile_idx = tl.program_id(0)
    attn_head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    # --- 2. 计算当前块的查询索引 ---
    start_query_idx = query_seq_tile_idx * BLOCK_Q
    query_offsets = start_query_idx + tl.arange(0, BLOCK_Q)

    # --- 3. 创建查询掩码 ---
    query_mask = query_offsets < query_seq_len

    # --- 4. 计算行起始指针 ---
    row_start_ptr = (
        input_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    # --- 5. Pass 1: 计算 Max 和 Sum (数值稳定softmax) ---
    m = tl.full([BLOCK_Q], -float("inf"), dtype=tl.float32)
    exp_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        block_ptr = row_start_ptr[:, None] + k_offsets[None, :]

        row_mask = query_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        mask = row_mask & col_mask

        s_block = tl.load(
            block_ptr, mask=mask, other=-float("inf"), cache_modifier=".ca"
        )
        s_block = s_block * scale_factor  # 应用缩放

        # 更新最大值
        m_new = tl.max(s_block, axis=1)
        m_old = m
        m = tl.maximum(m_old, m_new)

        # 更新指数和
        s_prev = tl.exp(m_old - m)
        exp_sum = exp_sum * s_prev

        s_curr = tl.exp(s_block - m[:, None])
        l_new = tl.sum(tl.where(mask, s_curr, 0.0), axis=1)
        exp_sum = exp_sum + l_new

    # --- 6. Pass 2: 计算并存储最终概率 ---
    exp_sum_inv = 1.0 / exp_sum

    out_row_start_ptr = (
        output_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)

        block_ptr_in = row_start_ptr[:, None] + k_offsets[None, :]
        block_ptr_out = out_row_start_ptr[:, None] + k_offsets[None, :]

        row_mask = query_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        mask = row_mask & col_mask

        s_block = tl.load(
            block_ptr_in, mask=mask, other=-float("inf"), eviction_policy="evict_first"
        )

        # 计算softmax概率
        s_block = s_block * scale_factor
        s_block = s_block - m[:, None]
        p_block = tl.exp(s_block)
        p_block = p_block * exp_sum_inv[:, None]

        tl.store(block_ptr_out, p_block, mask=mask, cache_modifier=".cs")


def scaled_softmax_forward(input_t: torch.Tensor, scale_factor: float):
    """
    Fused operation: output = softmax(input * scale_factor)

    Args:
        input_t: 输入张量 [B, H, Q_len, K_len]
        scale_factor: 缩放因子

    Returns:
        output_t: 输出张量 [B, H, Q_len, K_len]
    """
    # --- 参数校验 ---
    assert input_t.dim() == 4, "输入张量必须是4维的 [B, H, Q, K]"
    assert input_t.dtype in [torch.float16, torch.bfloat16], "仅支持 fp16 和 bf16 数据类型"

    batch_size, attn_heads, query_seq_len, key_seq_len = input_t.shape
    assert key_seq_len % 8 == 0, "键序列长度必须能被8整除"
    assert query_seq_len > 1, "查询序列长度必须大于1"

    # --- 计算 Grid 大小 ---
    def grid(meta):
        # 动态计算BLOCK_Q，基于key_seq_len选择合适的配置
        BLOCK_Q = meta["BLOCK_Q"]
        query_seq_tile_len = triton.cdiv(query_seq_len, BLOCK_Q)
        return (query_seq_tile_len, attn_heads, batch_size)

    # 创建输出张量
    output_t = torch.empty_like(input_t)

    # --- 获取内存步长 ---
    stride_b = input_t.stride(0)
    stride_h = input_t.stride(1)
    stride_q = input_t.stride(2)

    # --- 启动前向传播 Kernel ---
    scaled_softmax_forward_kernel[grid](
        output_t,
        input_t,
        scale_factor,
        query_seq_len,
        key_seq_len,
        stride_b,
        stride_h,
        stride_q,
    )
    return output_t


@libentry()
@triton.autotune(configs=autotune_configs, key=["query_seq_len", "key_seq_len"])
@triton.jit
def scaled_softmax_backward_kernel(
    grad_input_ptr,  # dS (输出梯度)
    grad_output_ptr,  # dP (输入梯度)
    output_ptr,  # P (前向输出)
    scale_factor,
    query_seq_len,
    key_seq_len,
    stride_b,
    stride_h,
    stride_q,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Scaled Softmax Backward Kernel
    实现: dS = scale * P * (dP - sum(P * dP))

    Args:
        grad_input_ptr: 输入梯度指针
        grad_output_ptr: 输出梯度指针
        output_ptr: 前向输出指针
        scale_factor: 缩放因子
        query_seq_len: 查询序列长度
        key_seq_len: 键序列长度
        stride_b: batch维度步长
        stride_h: head维度步长
        stride_q: query维度步长
        BLOCK_Q: Q维度分块大小
        BLOCK_K: K维度分块大小
    """

    # --- 1. 获取程序ID ---
    query_seq_tile_idx = tl.program_id(0)
    attn_head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    # --- 2. 计算当前块的查询索引 ---
    start_query_idx = query_seq_tile_idx * BLOCK_Q
    query_offsets = start_query_idx + tl.arange(0, BLOCK_Q)

    # --- 3. 创建查询掩码 ---
    query_mask = query_offsets < query_seq_len

    # --- 4. 计算所有张量的行起始指针 ---
    output_row_ptr = (
        output_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    grad_output_row_ptr = (
        grad_output_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    grad_input_row_ptr = (
        grad_input_ptr
        + batch_idx * stride_b
        + attn_head_idx * stride_h
        + query_offsets * stride_q
    )

    # --- 5. Pass 1: 计算 D = sum(P * dP) ---
    D = tl.zeros([BLOCK_Q], dtype=tl.float32)

    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = query_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        mask = row_mask & col_mask

        # 加载 P 和 dP
        ptr_P = output_row_ptr[:, None] + k_offsets[None, :]
        ptr_dP = grad_output_row_ptr[:, None] + k_offsets[None, :]

        P_block = tl.load(ptr_P, mask=mask, other=0.0, cache_modifier=".ca")
        dP_block = tl.load(ptr_dP, mask=mask, other=0.0, cache_modifier=".ca")

        # 累加点积
        dot_block = P_block * dP_block
        D += tl.sum(tl.where(mask, dot_block, 0.0), axis=1)

    # --- 6. Pass 2: 计算 dS = scale * P * (dP - D) ---
    for k_block_idx in range(0, tl.cdiv(key_seq_len, BLOCK_K)):
        k_offsets = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        row_mask = query_mask[:, None]
        col_mask = k_offsets[None, :] < key_seq_len
        mask = row_mask & col_mask

        # 加载 P 和 dP
        ptr_P = output_row_ptr[:, None] + k_offsets[None, :]
        ptr_dP = grad_output_row_ptr[:, None] + k_offsets[None, :]
        ptr_dS = grad_input_row_ptr[:, None] + k_offsets[None, :]

        P_block = tl.load(ptr_P, mask=mask, other=0.0, eviction_policy="evict_first")
        dP_block = tl.load(ptr_dP, mask=mask, other=0.0, eviction_policy="evict_first")

        # 计算梯度
        dZ_block = P_block * (dP_block - D[:, None])
        dS_block = scale_factor * dZ_block

        # 存储梯度
        tl.store(ptr_dS, dS_block, mask=mask, cache_modifier=".cs")


def scaled_softmax_backward(
    grad_output: torch.Tensor, softmax_results: torch.Tensor, scale_factor: float
):
    """
    Scaled Softmax 反向传播

    Args:
        grad_output: 上游梯度 [B, H, Q, K]
        softmax_results: 前向传播输出 [B, H, Q, K]
        scale_factor: 缩放因子

    Returns:
        grad_input: 输入梯度 [B, H, Q, K]
    """
    # 确保内存连续
    grad_output = grad_output.contiguous()
    softmax_results = softmax_results.contiguous()

    # --- 参数校验 ---
    assert grad_output.dim() == 4, "梯度张量必须是4维的"
    assert softmax_results.dim() == 4, "softmax结果必须是4维的"
    assert grad_output.dtype in [torch.float16, torch.bfloat16], "仅支持 fp16 和 bf16"
    assert softmax_results.dtype in [torch.float16, torch.bfloat16], "仅支持 fp16 和 bf16"

    batch_size, attn_heads, query_seq_len, key_seq_len = softmax_results.shape

    # --- 计算 Grid 大小 ---
    def grid(meta):
        # 动态计算BLOCK_Q，基于key_seq_len选择合适的配置
        BLOCK_Q = meta["BLOCK_Q"]
        query_seq_tile_len = triton.cdiv(query_seq_len, BLOCK_Q)
        return (query_seq_tile_len, attn_heads, batch_size)

    # 创建梯度输出张量
    grad_input = torch.empty_like(grad_output)

    # --- 获取内存步长 ---
    stride_b = softmax_results.stride(0)
    stride_h = softmax_results.stride(1)
    stride_q = softmax_results.stride(2)

    # --- 启动反向传播 Kernel ---
    scaled_softmax_backward_kernel[grid](
        grad_input,  # dS (输出)
        grad_output,  # dP (输入)
        softmax_results,  # P (输入)
        scale_factor,
        query_seq_len,
        key_seq_len,
        stride_b,
        stride_h,
        stride_q,
    )

    return grad_input
