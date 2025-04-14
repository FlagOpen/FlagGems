import torch

import flag_gems  # noqa: F401


def f(x, y):
    return torch.ops.flag_gems.add_tensor(x, y)


F = torch.compile(f)

x = torch.randn(2, 6, device="cuda:1")
y = torch.randn(6, device="cuda:1")
out = F(x, y)
ref = x + y
print(out)
print(ref)
