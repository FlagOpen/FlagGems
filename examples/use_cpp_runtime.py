import threading

import torch

import flag_gems.c_operators  # noqa: F401

x = torch.randn(10, device="cuda:0")
out = torch.ops.flag_gems.add_tensor(x, x)
print(out)

x = torch.randn(10, device="cuda:1")
out = torch.ops.flag_gems.add_tensor(x, x)
print(out)

x = torch.randn(10, device="cuda:2")
out = torch.ops.flag_gems.add_tensor(x, x)
print(out)


def f(x):
    print(torch.ops.flag_gems.add_tensor(x, x))


t = threading.Thread(target=f, args=(torch.randn(10, device="cuda:3"),))
t.start()
t.join()
