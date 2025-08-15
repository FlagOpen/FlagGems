import torch
import triton
import triton.language as tl


@triton.jit
def topkGatingSoftmax(
    input_ptr,
    finished_ptr,
    output_ptr,
    indices_ptr,
    source_rows_ptr,
    num_rows,
    k,
    start_expert,
    end_expert,
    num_experts,
    INDEX_TY: tl.constexpr,
    BLOCK_SIZE_E: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_active = True
    if finished_ptr is not None:
        row_active = tl.load(finished_ptr + row_id).to(tl.int1)

    if (row_id >= num_rows) or (not row_active):
        for k_idx in range(k):
            tl.store(output_ptr + row_id * k + k_idx, 0.0)
            tl.store(indices_ptr + row_id * k + k_idx, num_experts)  # 无效专家号
            tl.store(
                source_rows_ptr + row_id * k + k_idx,
                (k_idx * num_rows + row_id).to(tl.int32),
            )
        return

    cols = tl.arange(0, BLOCK_SIZE_E) + start_expert
    mask = (cols < end_expert) & (cols < num_experts)
    row_ptr = input_ptr + row_id * num_experts
    logits = tl.load(row_ptr + cols, mask=mask, other=-float("inf"))

    row_max = tl.max(logits, axis=0)
    logits = logits - row_max
    exp_vals = tl.exp(logits)
    row_sum = tl.sum(exp_vals, axis=0)
    probs = exp_vals / (row_sum + 1e-8)

    for k_idx in range(k):
        curr_max = tl.max(probs, axis=0)
        curr_arg = tl.argmax(probs, axis=0)

        tl.store(output_ptr + row_id * k + k_idx, curr_max)
        tl.store(indices_ptr + row_id * k + k_idx, curr_arg.to(INDEX_TY))
        tl.store(
            source_rows_ptr + row_id * k + k_idx,
            (k_idx * num_rows + row_id).to(tl.int32),
        )

        probs = tl.where(cols == curr_arg, -float("inf"), probs)


def topk_softmax(
    topk_weights: torch.Tensor,  # [num_tokens, topk]
    topk_indices: torch.Tensor,  # [num_tokens, topk]
    token_expert_indices: torch.Tensor,  # [num_tokens, topk]
    gating_output: torch.Tensor,  # [num_tokens, num_experts]
) -> None:
    assert gating_output.is_cuda and gating_output.dtype == torch.float32
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.size(-1)

    if topk_indices.dtype == torch.int32:
        index_ty = tl.int32
    elif topk_indices.dtype == torch.uint32:
        index_ty = tl.uint32
    elif topk_indices.dtype == torch.int64:
        index_ty = tl.int64
    else:
        raise TypeError("topk_indices must be int32/uint32/int64")

    BLOCK_SIZE_E = min(triton.next_power_of_2(num_experts), 4096)
    grid = (num_tokens,)

    topkGatingSoftmax[grid](
        gating_output,
        None,
        topk_weights,
        topk_indices,
        token_expert_indices,
        num_tokens,
        topk,
        0,
        num_experts,
        num_experts,
        INDEX_TY=index_ty,
        BLOCK_SIZE_E=BLOCK_SIZE_E,
    )
