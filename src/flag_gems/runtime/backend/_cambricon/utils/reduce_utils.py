import math

import triton
import triton.language as tl

from flag_gems import runtime

from . import MAX_NRAM_SIZE, TOTAL_CORE_NUM


def cfggen_reduce_op():
    return runtime.get_tuned_config("common_reduce_ops")


def cfggen_reduce_op2():
    block_size = [2048, 4096, 8192, 16384, 32768]
    num_stage = [1, 3]
    configs = [
        triton.Config(
            {"BLOCK_SIZE": m, "ITER_NUM": math.log2(m) + 1}, num_warps=1, num_stages=s
        )
        for m in block_size
        for s in num_stage
    ]
    return configs


def count_divisible_by_2(x):
    count = 0
    while x > 0 and x % 2 == 0:
        x //= 2
        count += 1
    return count


def next_power_of_two(x):
    if x < 16:
        return 16
    if x & (x - 1) == 0:
        return x
    return 1 << (x - 1).bit_length()


def prune_reduce_config(configs, named_args, **kwargs):
    M = named_args["M"]
    pruned_configs = []
    for config in configs:
        BLOCK_SIZE = config.kwargs["BLOCK_SIZE"]
        num_stages = config.num_stages
        num_block = M // BLOCK_SIZE
        if num_block < 1:
            continue
        if num_block < TOTAL_CORE_NUM:
            # A core must process a BLOCK_SIZE of data.
            if num_stages > 1:
                continue
            # The final IR will only have two allocs of BLOCK_SIZE:
            # - one for the pad generated by the mask load;
            # - one for for the dst of computation;
            alloc_num = 2
        else:
            # A core may process more than one BLOCK_SIZE of data.
            # The final IR will only have four allocs of BLOCK_SIZE:
            # - one for the _tmp to receive the value;
            # - one for the pad generated by the mask load;
            # - one for for the dst of computation;
            # - one for the return value of for.
            alloc_num = 4
        # Set f32 as the default type.
        if BLOCK_SIZE * 4 * alloc_num <= MAX_NRAM_SIZE:
            pruned_configs.append(config)
    # If M < 1024, append the default config.
    if len(pruned_configs) == 0:
        pruned_configs.append(
            triton.Config(
                {"BLOCK_SIZE": next_power_of_two(M)}, num_warps=1, num_stages=1
            )
        )
    return pruned_configs
