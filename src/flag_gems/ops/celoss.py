import logging

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 4, "BLOCK_C": 1024, "BLOCK_D": 2}, num_warps=4)
        # for n in [4, 8, 16]
        # for c in [1024, 2048]
        # for d in [1, 16, 64]
        # for w in [4, 8]
    ],
    key=["N", "C", "D"],
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
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + offset_n[:, None] * D + offset_d[None, :]
    tgt_mask = offset_n[:, None] < N and offset_d[None, :] < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)

    ignore_mask = tgt == ignore_index

    inp_tgt_ptrs = inp_ptr + offset_n[:, None] * C * D + tgt * D + offset_d[None, :]
    inp_tgt = tl.load(inp_tgt_ptrs, mask=tgt_mask, other=0).to(tl.float32)

    w_tgt_ptrs = w_ptr + tgt
    w_tgt = tl.load(w_tgt_ptrs, mask=tgt_mask, other=0).to(tl.float32)

    _sum = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = (
            inp_ptr
            + offset_n[:, None, None] * C * D
            + offset_c[None, :, None] * D
            + offset_d[None, None, :]
        )
        inp_mask = (
            offset_n[:, None, None] < N and offset_c[None, :, None] < C
        ) and offset_d[None, None, :] < D
        inp = tl.load(inp_ptrs, mask=inp_mask, other=0).to(tl.float32)
        sum_inp = tl.sum(inp, axis=1)
        _sum = _sum + sum_inp
        tl.device_print("sum: ", _sum)  # comment this line will cause result error

    # out = (tl.log(_sum) - log_inp) * w_tgt
    out = tl.log(_sum / inp_tgt) * w_tgt
    out = tl.where(ignore_mask, 0, out)
    out_ptrs = out_ptr + offset_n[:, None] * D + offset_d[None, :]
    tl.store(out_ptrs, out, mask=tgt_mask)


class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, weight, reduction, ignore_index, label_smoothing):
        logging.debug("GEMS CrossEntropyLoss")

        # suppose reduction == NONE
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

        inp = torch.exp(inp.contiguous())
        tgt = target.contiguous()
        weight = weight.contiguous()
        out = torch.empty(shape, dtype=inp.dtype, device=inp.device)

        grid = lambda meta: (
            triton.cdiv(N, meta["BLOCK_N"]),
            triton.cdiv(D, meta["BLOCK_D"]),
        )
        with torch.cuda.device(inp.device):
            celoss_kernel[grid](inp, tgt, weight, out, ignore_index, N, C, D)

        return out


def celoss(
    inp, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0
):
    return CrossEntropyLoss.apply(
        inp, target, weight, reduction, ignore_index, label_smoothing
    )


BATCH = 4
SIZE = [i * 4 for i in range(1, 22, 5)]

for size in SIZE:
    shape = [BATCH, size, 2]
    dtype = torch.float32
    up_limit = size
    target_shape = [BATCH, 2]

    weight = torch.randn(
        [
            size,
        ],
        dtype=dtype,
        device="cuda",
    )
    loss = torch.nn.CrossEntropyLoss(weight=weight, reduction="none", ignore_index=-100)
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    tgt = torch.randint(0, 3, target_shape, device="cuda")
    out_ = loss(inp, tgt)
    out = celoss(inp, tgt, weight=weight, ignore_index=-100)
    flag = torch.allclose(out_, out)
    print(f"SHAPE {shape}: {flag}")
    # import pdb; pdb.set_trace()
    # latency_torch = triton.testing.do_bench(
    #     lambda: loss(inp, tgt),
    #     warmup=100,
    #     rep=1000,
    #     return_mode="median"
    # )
    # latency_triton = triton.testing.do_bench(
    #     lambda: celoss(inp, tgt),
    #     warmup=100,
    #     rep=1000,
    #     return_mode="median"
    # )
    # print(f"TORCH COST: {latency_torch}, TRITON COST: {latency_triton}")
