import logging

import torch

# import triton
# import triton.language as tl


def multinomial(prob, n_samples, with_replacement=False, *, gen=None):
    logging.debug("GEMS MULTINOMIAL")
    assert prob.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    assert 0 < prob.dim() <= 2, "prob_dist must be 1 or 2 dim"
    n_categories = prob.size(-1)
    assert n_categories <= (1 << 24), "number of categories cannot exceed 2^24"
    assert (
        with_replacement or n_samples <= n_categories
    ), "cannot sample n_samples > prob.size(-1) samples without replacement."
    out_size = (prob.size(0), n_samples) if prob.dim() == 2 else (n_samples,)
    out = torch.empty(out_size, device=prob.device, dtype=torch.int64)

    if (not with_replacement) or n_samples == 1:
        # s = argmax( p / q ) where q ~ Exp(1)
        q = torch.empty_like(prob).exponential_(1.0)
        s = torch.div(prob, q, out=q)
        if n_samples == 1:
            return torch.argmax(s, dim=-1)
        else:
            vals, indices = torch.topk(s, n_samples, dim=-1)
            return indices

    # grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    return out
