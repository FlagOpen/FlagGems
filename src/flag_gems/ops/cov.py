import logging
import torch
import triton
import triton.language as tl
import triton.testing

@triton.jit
def mean_kernel(
    X, 
    mean, 
    M, 
    N, 
    weights,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    cols = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0)
    w = tl.load(weights + cols, mask=mask, other=1.0) 
    weighted_x = x * w
    sum_x = tl.sum(weighted_x)
    tl.atomic_add(mean + row, sum_x)

@triton.jit
def covariance_kernel(
    X, 
    cov_matrix, 
    mean, 
    M, 
    N, 
    weights,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    block_id = tl.program_id(2)
    cols = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_row = tl.load(X + row * N + cols, mask=mask, other=0.0) - tl.load(mean + row)
    x_col = tl.load(X + col * N + cols, mask=mask, other=0.0) - tl.load(mean + col)

    w = tl.load(weights + cols, mask=mask, other=1.0)
    cov = tl.sum(w * x_row * x_col)
    tl.atomic_add(cov_matrix + row * M + col, cov)

def cov(X, correction=1, fweights=None, aweights=None):
    logging.debug("GEMS COV")    
    M, N = X.shape  
    if fweights is not None:
        fweights = fweights.to(device=X.device, dtype=X.dtype)
    else:
        fweights = torch.ones(N, device=X.device, dtype=X.dtype)

    if aweights is not None:
        aweights = aweights.to(device=X.device, dtype=X.dtype)
    else:
        aweights = torch.ones(N, device=X.device, dtype=X.dtype)

    weights = fweights * aweights
    total_weight = weights.sum()
    sum_wi_ai = (weights * aweights).sum() 

    adjustment = (sum_wi_ai / total_weight) * correction if correction != 0 else 0

    denominator = torch.clamp(total_weight - adjustment, min=0)
    if denominator <= 0:
        raise ValueError("Non-positive denominator in covariance calculation.")
    
    mean = torch.zeros(M, device=X.device, dtype=X.dtype)
    cov_matrix = torch.zeros((M, M), device=X.device, dtype=X.dtype)

    BLOCK_SIZE = min(256, N)
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    grid = lambda meta: (M, num_blocks)
    mean_kernel[grid](X, mean, M, N, weights, BLOCK_SIZE=BLOCK_SIZE)
    mean = mean / total_weight

    grid_cov = lambda meta: (M, M, num_blocks)
    covariance_kernel[grid_cov](X, cov_matrix, mean, M, N, weights, BLOCK_SIZE=BLOCK_SIZE)
    cov_matrix = cov_matrix / denominator

    return cov_matrix