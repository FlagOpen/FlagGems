import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_cuda_seed_offset, uniform


@libentry()
@triton.heuristics(
    {
        "NBLOCK": lambda args: 128,
        "num_warps": lambda args: 4,
    }
)
@triton.jit(do_not_specialize=["K", "N", "philox_seed", "philox_offset"])
def multinomial_with_replacement(
    cdf_ptr, out_ptr, K, N, philox_seed, philox_offset, NBLOCK: tl.constexpr
):
    # The computation is arranged in a 2d grid of blocks, each producing
    # a batch of samples for a particular distribution.
    #            <------------------- grid.x --------------------->
    #           |   dist0.batch0 | dist0.batch1 | dist0.batch2 ...
    #   grid.y  |   dist1.batch0 | dist1.batch1 | dist1.batch2 ...
    #           |   dist2.batch0 | dist2.batch1 | dist2.batch2 ...
    y_off = tl.program_id(1) * N
    n = tl.program_id(0) * NBLOCK + tl.arange(0, NBLOCK)
    rv, _, _, _ = uniform(philox_seed, philox_offset, y_off + n)

    # Do a binary search for each random variable on the cdf
    start = tl.zeros((NBLOCK,), dtype=tl.int32)
    end = tl.zeros((NBLOCK,), dtype=tl.int32) + K
    steps = tl.math.log2(K.to(tl.float32)).to(tl.int32)
    cdf_ptr += tl.program_id(1) * K
    for _ in range(steps):
        mid = start + (end - start) // 2
        x = tl.load(cdf_ptr + mid, mask=n < N)
        start = tl.where(x < rv, mid + 1, start)
        end = tl.where(x < rv, end, mid)

    # Returns the last index in case of an overflow
    start = tl.where(start == K, start - 1, start)

    y_off = tl.program_id(1) * N
    out_ptr += y_off
    tl.store(out_ptr + n, start, mask=n < N)


def multinomial(prob, n_samples, with_replacement=False, *, gen=None):
    logging.debug("GEMS MULTINOMIAL")
    assert prob.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    assert 0 < prob.dim() <= 2, "prob_dist must be 1 or 2 dim"
    n_categories = prob.size(-1)
    assert n_categories <= (1 << 24), "number of categories cannot exceed 2^24"
    assert (
        with_replacement or n_samples <= n_categories
    ), "cannot sample n_samples > prob.size(-1) samples without replacement."

    # Sampling without replacement
    if (not with_replacement) or n_samples == 1:
        # s = argmax( p / q ) where q ~ Exp(1)
        q = torch.empty_like(prob).exponential_(1.0)
        s = torch.div(prob, q, out=q)
        if n_samples == 1:
            return torch.argmax(s, dim=-1)
        else:
            vals, indices = torch.topk(s, n_samples, dim=-1)
            return indices

    # Sampling with replacement
    normed_prob = torch.empty_like(prob, memory_format=torch.contiguous_format)
    normed_prob.copy_(prob)
    if normed_prob.dim() == 1:
        normed_prob = normed_prob.view(1, -1)
    normed_prob = torch.div(
        normed_prob, normed_prob.sum(-1, keepdim=True), out=normed_prob
    )
    cdf = torch.cumsum(normed_prob, -1, out=normed_prob)
    n_dist = normed_prob.size(0)
    out = torch.empty((n_dist, n_samples), device=prob.device, dtype=torch.int32)
    # The CTA level parallelism is framed in a 2d grid of blocks with grid.y
    # indexing into distributions and grid.x output sample batches
    increment = n_dist * n_samples
    philox_seed, philox_offset = philox_cuda_seed_offset(increment)
    grid = lambda META: (triton.cdiv(n_samples, META["NBLOCK"]), n_dist)
    multinomial_with_replacement[grid](
        cdf, out, n_categories, n_samples, philox_seed, philox_offset
    )
    return out
