import torch

import flag_gems

# 创建一个形状为 (3, 4) 的张量 x
shape_x = (3, 4)
x = torch.randn(shape_x, device=flag_gems.device)

# 创建一个形状为 (1, 4) 的张量 y
shape_y = (1, 4)
y = torch.randn(shape_y, device=flag_gems.device)

# 使用 flag_gems 的上下文
with flag_gems.use_gems():
    # 这里的 y 将会被广播到形状 (3, 4) 以匹配 x 的形状
    C = torch.add(x, y)

print("x:", x)
print("y:", y)
print("C:", C)
