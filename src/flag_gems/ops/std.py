import logging

import torch

from .sqrt import sqrt
from .var_mean import var_mean

logger = logging.getLogger(__name__)


def std(x, dim=None, unbiased=True, keepdim=False):
    logger.debug("GEMS STD Forward")

    dim_list = dim
    if isinstance(dim, int):
        dim_list = [dim]

    variance, mean = var_mean(x, dim=dim_list, unbiased=unbiased, keepdim=keepdim)

    std_dev = sqrt(variance)

    return std_dev, mean


def std_backward(grad_output, std_dev, x, mean, dim, unbiased):
    logger.debug("GEMS STD Backward")

    dim_list = dim
    if isinstance(dim, int):
        dim_list = [dim]

    if dim_list is None:
        num_elements = x.numel()
    else:
        num_elements = 1
        for d in dim_list:
            num_elements *= x.shape[d]

    divisor = num_elements - 1.0 if unbiased else num_elements

    std_dev_safe = std_dev.where(std_dev > 0, torch.ones_like(std_dev))

    if grad_output.dim() < x.dim():
        if dim_list is None:
            grad_output = grad_output.reshape([1] * x.dim())
        else:
            grad_output = grad_output.unsqueeze(dim_list)

    if std_dev.dim() < x.dim():
        if dim_list is None:
            std_dev_safe = std_dev_safe.reshape([1] * x.dim())
        else:
            std_dev_safe = std_dev_safe.unsqueeze(dim_list)

    if mean.dim() < x.dim():
        if dim_list is None:
            mean = mean.reshape([1] * x.dim())
        else:
            mean = mean.unsqueeze(dim_list)

    grad_x = grad_output * (1.0 / std_dev_safe) * (x - mean) / divisor

    return grad_x
