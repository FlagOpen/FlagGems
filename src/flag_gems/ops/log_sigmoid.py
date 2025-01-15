import logging

import triton
import triton.language as tl

from ..utils import pointwise_dynamic


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def log_sigmoid_forward(x):
    max_val = tl.maximum(-x, 0.0)
    exp_max_val = tl.exp(-max_val)
    exp_x_max_val = tl.exp(-x - max_val)

    return -max_val - tl.log(exp_max_val + exp_x_max_val)


def log_sigmoid(x):
    logging.debug("GEMS LOG_SIGMOID FORWARD")

    return log_sigmoid_forward(x)
