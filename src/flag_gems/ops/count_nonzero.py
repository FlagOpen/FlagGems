import torch
import triton
import triton.language as tl


def cfggen():
    BLOCK_SIZE = [1024,2048,4096]
    configs = [
        triton.Config({"BLOCK_SIZE": block_size}, num_warps=4)
        for block_size in BLOCK_SIZE
    ]
    return configs

@triton.jit
def count_nonzero_kernel_1(
    x_ptr,         # 输入张量指针
    out_ptr,       # 输出结果指针
    numel,         # 张量总元素数
    BLOCK_SIZE: tl.constexpr  # 每个线程块处理的元素数
):
    # 线程块的起始位置
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # 当前线程的索引
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 确保访问在张量范围内
    mask = offsets < numel

    # 加载输入数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # 计算非零元素：x != 0
    is_nonzero = (x != 0).to(tl.int32)
    nonzero_count = tl.sum(is_nonzero, axis=0)

    # 使用原子加将结果写入到输出张量
    tl.atomic_add(out_ptr, nonzero_count)


@triton.autotune(configs=cfggen(), key=["numel"])
@triton.jit
def count_nonzero_kernel(
    x_ptr,          # 输入张量指针
    out_ptr,        # 输出结果指针
    N,
    numel,          # 张量总元素数
    BLOCK_SIZE: tl.constexpr  # 每个线程块处理的元素数
):
    # 线程块的起始位置
    pid_x = tl.program_id(0)

    nonzero_count = tl.full((),value=0,dtype=out_ptr.dtype.element_ty)
    for start_n in range(0,N,BLOCK_SIZE):
        # 当前线程的索引
        cols_offsets =  start_n + tl.arange(0, BLOCK_SIZE)
        offset = pid_x * N + cols_offsets
        # 确保访问在张量范围内
        mask = offset <  numel and cols_offsets < N
        # 加载输入数据
        x = tl.load(x_ptr + offset, mask=mask, other=0)
        # 计算非零元素
        is_nonzero = (x != 0).to(tl.int32)  # 将布尔值转换为整型
        nonzero_count += tl.sum(is_nonzero)
        
    # 使用原子加将结果写入到输出张量
    tl.store(out_ptr + pid_x, nonzero_count)
     


def dim_compress(inp, dims):
    #如果传入的维度是int类型的话，转换为列表形式
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim #获取传入张量的总维度数
    stride = inp.stride() #获取各个维度的步长 返回的是一个元组
    batch_dim = [i for i in range(dim) if i not in dims] #所有不在 dims 列表中的维度索引
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()


def count_nonzero(x,dim=None):
    """
    Triton 实现的 count_nonzero 函数。
    参数:
        x: 输入张量 (torch.Tensor)
    返回值:
        非零元素的总数量 (int)
    """
    if dim is not None:
        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        #print(torch.count_nonzero(x,dim))
        shape = x.shape
        out_shape = list(shape)
        del out_shape[dim]
        out = torch.zeros(out_shape, dtype=torch.int32, device=x.device)
        #TILE_N = triton.next_power_of_2(shape[dim])
        numel = x.numel()
        grid = lambda meta: (triton.cdiv(numel, shape[dim]),)
        x = dim_compress(x,dim)
        # 确保输入张量是连续的
        x = x.contiguous().flatten()
        # 调用 Triton 内核
        count_nonzero_kernel[grid](
            x,  
            out,
            shape[dim],
            numel
        )

        # 读取结果并返回
        return out
    else:
        # 将输入张量展平以便处理
        x = x.contiguous().flatten()
        numel = x.numel()
        print("走这里了吗")
        # 初始化输出结果张量（单个元素，存储最终结果）
        out = torch.zeros(1, dtype=torch.int32, device=x.device)

        # Triton 的网格大小和线程块大小
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)

        # 调用 Triton 内核
        count_nonzero_kernel_1[grid](
            x,
            out,
            numel,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # 读取结果并返回
        return out[0]


