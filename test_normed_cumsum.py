import torch

import flag_gems

torch.set_printoptions(precision=6)
x = torch.ones(10000, 128, device="cuda")
print(x)
x_cums = flag_gems.ops.normed_cumsum(x, dim=-1)
print(x_cums)
