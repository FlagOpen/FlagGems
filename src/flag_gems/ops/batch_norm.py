# import logging

import torch

# import triton


class BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        cudnn_enable,
    ):
        print("GEMS BATCHNORM FORWARD")
        return input

    @staticmethod
    def backward(ctx, output_grad):
        print("GEMS BATCHNORM BACKWARD")
        return (None,) * 9


def batch_norm(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
    cudnn_enable=True,
):
    return BatchNorm.apply(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        cudnn_enable,
    )
