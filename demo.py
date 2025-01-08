import torch
import flag_gems

# 测试
x = -torch.ones(10, device='cuda') * 100

#使用 Triton 实现的 log_sigmoid
triton_result = flag_gems.log_sigmoid(x)
torch_result = torch.nn.functional.logsigmoid(x)
# 比较结果
print("x (All Negative):\n", x)
print("Triton result:\n", triton_result)
print("PyTorch result:\n", torch_result)
# 计算差异
difference = torch.abs(triton_result - torch_result)
print("Difference:\n", difference)
#with flag_gems.use_gems():
#    y = torch.log_sigmoid(x)
#print(y)