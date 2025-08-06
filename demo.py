import torch

import flag_gems

shape = (4,)
x = torch.randn(shape, device=flag_gems.device)
y = torch.randn_like(x)
with flag_gems.use_gems():
    C = torch.add(x, y)
