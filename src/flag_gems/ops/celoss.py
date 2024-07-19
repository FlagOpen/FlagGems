import logging

import torch
import triton
import triton.language as tl

from flag_gems import sum


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": n}, num_warps=4) for n in [512, 1024, 2048]],
    key=["N"],
)
@triton.jit
def weight_kernel(w_ptr, tgt_ptr, w_tgt_ptr, ignore_index, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    tgt_ptrs = tgt_ptr + offset

    tgt = tl.load(tgt_ptrs, mask, other=0)
    ignore_mask = tgt == ignore_index

    w_ptrs = w_ptr + tgt
    w = tl.load(w_ptrs, mask, other=0)
    w_tgt = tl.where(ignore_mask, 0, w)

    w_tgt_ptrs = w_tgt_ptr + offset
    tl.store(w_tgt_ptrs, w_tgt, mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": c, "BLOCK_D": d}, num_warps=4)
        for c in [256, 512, 1024]
        for d in [1, 4, 16]
    ],
    key=["C", "D"],
)
@triton.jit
def celoss_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    ignore_index,
    N,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)

    ignore_mask = tgt == ignore_index

    w_tgt_ptrs = w_ptr + pid_n * D + offset_d
    w_tgt = tl.load(w_tgt_ptrs, mask=tgt_mask, other=0).to(tl.float32)

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, inp_mask, other=-float("inf")).to(tl.float32)
        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=0)
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max[None, :])
    final_sum = tl.sum(tmp_sum, axis=0)

    inp_tgt_ptrs = inp_ptr + pid_n * C * D + tgt * D + offset_d
    inp_tgt = tl.load(inp_tgt_ptrs, mask=tgt_mask, other=-float("inf")).to(tl.float32)
    exp_tgt = tl.exp(inp_tgt - final_max)

    out = tl.log(final_sum / exp_tgt) * w_tgt
    out = tl.where(ignore_mask, 0, out)
    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=tgt_mask)


class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, weight, reduction, ignore_index, label_smoothing):
        logging.debug("GEMS CrossEntropyLoss")
        # label_smoothing not supported

        shape = list(inp.shape)
        dim = inp.ndim
        N = 1 if dim == 1 else shape[0]
        C = shape[0] if dim == 1 else shape[1]
        D = inp.numel() // N // C
        axis = 0 if dim == 1 else 1
        del shape[axis]

        if weight is None:
            weight = torch.ones(
                [
                    C,
                ],
                dtype=inp.dtype,
                device=inp.device,
            )

        inp = inp.contiguous()
        tgt = target.contiguous()
        weight = weight.contiguous()
        w_tgt = torch.empty(shape, dtype=inp.dtype, device=inp.device)
        out = torch.empty(shape, dtype=inp.dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(N * D, meta["BLOCK_SIZE"]),)
        with torch.cuda.device(inp.device):
            weight_kernel[grid](weight, tgt, w_tgt, ignore_index, N * D)

        grid = lambda meta: (
            N,
            triton.cdiv(D, meta["BLOCK_D"]),
        )
        with torch.cuda.device(inp.device):
            celoss_kernel[grid](inp, tgt, w_tgt, out, ignore_index, N, C, D)

        if reduction == 0:  # NONE
            return out
        elif reduction == 1:  # MEAN
            return sum(out) / sum(w_tgt)
        else:  # SUM
            return sum(out)


def celoss(
    inp, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0
):
    return CrossEntropyLoss.apply(
        inp, target, weight, reduction, ignore_index, label_smoothing
    )


BATCH = 1024
SIZE = [i * 64 for i in range(1, 21)]

for size in SIZE:
    shape = [BATCH, size, 64]
    dtype = torch.float32
    up_limit = size
    target_shape = [BATCH, 64]

    weight = torch.randn(
        [
            size,
        ],
        dtype=dtype,
        device="cuda",
    )
    loss = torch.nn.CrossEntropyLoss(weight=weight, reduction="sum", ignore_index=-100)
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    tgt = torch.randint(0, 3, target_shape, device="cuda")
    out_ = loss(inp, tgt)
    out = celoss(inp, tgt, weight=weight, ignore_index=-100, reduction=2)
    flag = torch.allclose(out_, out)
    diff = torch.max(torch.abs(out_ - out))
    print(f"SHAPE {shape}: {flag}")
    # print(f"MAX DIFF: {diff}")
    # import pdb; pdb.set_trace()
    latency_torch = triton.testing.do_bench(
        lambda: loss(inp, tgt), warmup=100, rep=1000, return_mode="median"
    )
    latency_triton = triton.testing.do_bench(
        lambda: celoss(inp, tgt, weight=weight, ignore_index=-100, reduction=2),
        warmup=100,
        rep=1000,
        return_mode="median",
    )
    print(f"TORCH COST: {latency_torch}, TRITON COST: {latency_triton}")
