import logging
import torch
import triton
import triton.language as tl
import triton.testing

MAX_GRID_NUM = 65535

@triton.jit
def mean_kernel(
    X, 
    mean, 
    M, 
    N, 
    weights,
    row_offset: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0) + row_offset  
    if row >= M:
        return 
    
    acc = 0.0
    for block_start in range(0, N, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask=mask, other=0.0)
        w = tl.load(weights + cols, mask=mask, other=0.0)
        acc += tl.sum(x * w, axis=0)
    tl.atomic_add(mean + row, acc)

@triton.jit
def covariance_kernel(
    X, 
    cov_matrix, 
    mean, 
    M, 
    N, 
    weights,
    row_offset: tl.constexpr,
    col_offset: tl.constexpr,  
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0) + row_offset
    col = tl.program_id(1) + col_offset
    if row >= M or col >= M:
        return
    
    acc = 0.0
    mean_row = tl.load(mean + row)
    mean_col = tl.load(mean + col)
    
    for block_start in range(0, N, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_row = tl.load(X + row * N + cols, mask=mask, other=0.0)
        x_col = tl.load(X + col * N + cols, mask=mask, other=0.0)
        w = tl.load(weights + cols, mask=mask, other=0.0)
        x_row_centered = x_row - mean_row
        x_col_centered = x_col - mean_col
        acc += tl.sum(w * x_row_centered * x_col_centered, axis=0)
    tl.atomic_add(cov_matrix + row * M + col, acc)

def cov(X, correction=1, fweights=None, aweights=None):
    logging.debug("GEMS COV")    
    M, N = X.shape  
    
    if fweights is None:
        fweights = torch.ones(N, device=X.device, dtype=X.dtype)
    else:
        fweights = fweights.to(device=X.device, dtype=X.dtype)
    if aweights is None:
        aweights = torch.ones(N, device=X.device, dtype=X.dtype)
    else:
        aweights = aweights.to(device=X.device, dtype=X.dtype)
    weights = fweights * aweights
    sum_weights = weights.sum()
    sum_wa = (weights * aweights).sum()
    
    adjustment = (sum_wa / sum_weights) * correction if correction != 0 else 0
    denominator = torch.clamp(sum_weights - adjustment, min=0)    
    if denominator <= 0:
        raise ValueError("Non-positive denominator in covariance calculation.")
    
    mean = torch.zeros(M, device=X.device, dtype=X.dtype)
    cov_matrix = torch.zeros((M, M), device=X.device, dtype=X.dtype)

    BLOCK_SIZE = min(128, triton.next_power_of_2(N))

    for i in range((M + MAX_GRID_NUM - 1) // MAX_GRID_NUM):
        row_offset = i * MAX_GRID_NUM
        current_M = min(MAX_GRID_NUM, M - row_offset)
        grid = (current_M,)
        mean_kernel[grid](X, mean, M, N, weights, row_offset=row_offset, BLOCK_SIZE=BLOCK_SIZE)
    mean = mean / sum_weights
         
    for i in range((M + MAX_GRID_NUM - 1) // MAX_GRID_NUM):
        row_offset = i * MAX_GRID_NUM
        current_rows = min(MAX_GRID_NUM, M - row_offset)    
        for j in range((M + MAX_GRID_NUM - 1) // MAX_GRID_NUM):
            col_offset = j * MAX_GRID_NUM
            current_cols = min(MAX_GRID_NUM, M - col_offset)
            grid = (current_rows, current_cols)
            covariance_kernel[grid](X, cov_matrix, mean, M, N, weights, row_offset=row_offset, col_offset=col_offset, BLOCK_SIZE=BLOCK_SIZE)
    cov_matrix = cov_matrix / denominator
    return cov_matrix