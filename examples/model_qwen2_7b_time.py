import torch
import timeit
import flag_gems

# 配置参数
REPEAT = 10   # 总重复次数
NUMBER = 100   # 每次计时的运行次数

# 准备测试输入
inp = torch.empty((5), device='cpu', dtype=torch.float16)

# ---- 正确性验证 ----
with flag_gems.use_gems():  # 直接使用上下文管理器
    res_out = torch.ones_like(inp)
ref_out = torch.ones_like(inp)

print("[正确性验证]")
print("FlagGems输出:\n", res_out)
print("PyTorch原生输出:\n", ref_out)
print("结果一致性:", torch.allclose(res_out, ref_out, atol=5e-3, rtol=0))

# ---- 修复后的性能测试 ----
def benchmark_operator(use_gems: bool):
    # 定义可配置的上下文管理器
    def get_context():
        if use_gems:
            return flag_gems.use_gems()
        else:
            return torch.no_grad()
    
    # 确保每次重新创建上下文
    def wrapper():
        with get_context():  # 每次生成新实例
            return torch.ones_like(inp)
    # 精确计时
    timer = timeit.Timer(wrapper)
    times = timer.repeat(repeat=REPEAT, number=NUMBER)
    
    return sorted(times)[REPEAT//2] * 1e6 / NUMBER

# 执行测试
custom_time = benchmark_operator(use_gems=True)
native_time = benchmark_operator(use_gems=False)

print("\n[性能对比]")
print(f"FlagGems平均耗时: {custom_time:.3f} μs/次")
print(f"原生实现平均耗时: {native_time:.3f} μs/次")
print(f"性能差异: {abs(custom_time-native_time)/native_time*100:.1f}%")