import logging

from .sqrt import sqrt as gems_sqrt
from .var_mean import var_mean

logger = logging.getLogger(__name__)


def std(x, dim=None, unbiased=True, keepdim=False):
    logger.debug("GEMS STD Forward")

    dim_list = (
        dim if isinstance(dim, (list, tuple)) else ([dim] if dim is not None else None)
    )

    variance, _ = var_mean(x, dim=dim_list, unbiased=unbiased, keepdim=keepdim)

    std_dev = gems_sqrt(variance)

    return std_dev
