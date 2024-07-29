import torch
import triton
import triton.language as tl


@triton.jit
def triton_pad(
    x_ptr,
    out_ptr,
    in_dim0,
    in_dim1,
    in_strides0,
    in_strides1,
    out_strides0,
    out_strides1,
    valid_dim0_start,
    valid_dim0_end,
    valid_dim1_start,
    valid_dim1_end,
    in_elem_cnt: tl.constexpr,
    out_elem_cnt: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    block_offset = pid * BLOCK_SIZE

    offset = block_offset + tl.arange(0, BLOCK_SIZE)

    remaining = offset
    idx = remaining // out_strides0
    dst_index_0 = idx
    remaining = remaining - idx * out_strides0

    idx = remaining // out_strides1
    dst_index_1 = idx

    if_pad_false_mask = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    if_pad_true_mask = tl.full((BLOCK_SIZE,), 1, dtype=tl.int32)

    src_index_0 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    src_index_1 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    cond = dst_index_0 >= valid_dim0_start and dst_index_0 < valid_dim0_end
    cond &= dst_index_1 >= valid_dim1_start and dst_index_1 < valid_dim1_end
    if_pad = tl.where(cond, if_pad_false_mask, if_pad_true_mask).to(tl.int1)

    src_index_0 = dst_index_0 - valid_dim0_start
    src_index_1 = dst_index_1 - valid_dim1_start

    src_index_0 = tl.where(src_index_0 < 0, 0, src_index_0)
    src_index_1 = tl.where(src_index_1 < 0, 0, src_index_1)

    src_offset = src_index_0 * in_strides0 + src_index_1 * in_strides1
    load_cond = src_offset < in_elem_cnt
    x_val = tl.load(x_ptr + src_offset, mask=(not if_pad) and load_cond, other=0.0)
    tl.store(out_ptr + offset, x_val, mask=offset < out_elem_cnt)


def test_triton_pad(x, pad):
    ndim = x.ndim
    pad_size = len(pad)
    assert pad_size % 2 == 0

    pad_before = [0 for _ in range(ndim)]
    pad_after = [0 for _ in range(ndim)]

    pad_pair = pad_size // 2
    for i in range(pad_pair):
        pad_before[ndim - i - 1] = pad[2 * i]
        pad_after[ndim - i - 1] = pad[2 * i + 1]

    dst_shape = list(x.shape)
    for i in range(ndim):
        dst_shape[i] += pad_before[i] + pad_after[i]

    out = torch.empty(dst_shape, device=x.device, dtype=x.dtype)

    valid_dim0_start = pad_before[0]
    valid_dim0_end = dst_shape[0] - pad_before[0]

    valid_dim1_start = pad_before[1]
    valid_dim1_end = dst_shape[1] - pad_before[1]

    BLOCK_SIZE = 256
    grid = triton.cdiv(out.numel(), BLOCK_SIZE)
    triton_pad[grid,](
        x,
        out,
        x.shape[0],
        x.shape[1],
        x.stride()[0],
        x.stride()[1],
        out.stride()[0],
        out.stride()[1],
        valid_dim0_start,
        valid_dim0_end,
        valid_dim1_start,
        valid_dim1_end,
        x.numel(),
        out.numel(),
        BLOCK_SIZE,
        # num_warps=1,
    )
    print("triton out is: ", out)


#     def triton_pad(
#     x_ptr,
#     out_ptr,
#     in_dim0,
#     in_dim1,
#     in_strides0,
#     in_strides1,
#     out_strides0,
#     out_strides1,
#     valid_dim0_start,
#     valid_dim0_end,
#     valid_dim1_start,
#     valid_dim1_end,
#     in_elem_cnt,
#     out_elem_cnt,
#     BLOCK_SIZE: tl.constexpr,
# )

# x = torch.ones((4, 4), device="cuda", dtype=torch.float32)
x = torch.ones((4096, 4096), device="cuda", dtype=torch.float32)

pad_params = (2, 2)
# pad_x = torch.nn.functional.pad(x, (2, 2, 2, 2))
pad_x = torch.nn.functional.pad(x, pad_params)
print("Pad x is: ", pad_x)

test_triton_pad(x, pad_params)
