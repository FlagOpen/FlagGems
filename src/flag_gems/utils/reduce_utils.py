import math
import triton
import triton.language as tl

def cfggen_reduce_op():
    block_size = [64, 128, 512, 1024, 2048, 4096]
    num_stage = [1, 2]
    configs=[
        triton.Config({"BLOCK_SIZE": m}, num_warps=1, num_stages=s) for m in block_size for s in num_stage
    ]
    return configs


def cfggen_reduce_op2():
    block_size = [64, 128, 512, 1024, 2048, 4096]
    num_stage = [1, 2]
    configs=[
        triton.Config({"BLOCK_SIZE": m, "ITER_NUM": math.log2(m) + 1}, num_warps=1, num_stages=s) for m in block_size for s in num_stage
    ]
    return configs

def count_divisible_by_2(x):
    count = 0
    while x > 0 and x % 2 == 0:
        x //= 2
        count += 1
    return count