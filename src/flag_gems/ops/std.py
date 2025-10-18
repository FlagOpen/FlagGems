import logging

from .sqrt import sqrt as gems_sqrt
from .var_mean import var_mean

logger = logging.getLogger(__name__)


def std(x, dim=None, unbiased=True, keepdim=False):
    logger.debug("GEMS STD Forward")

    correction = int(unbiased)

    variance, _ = var_mean(x, dim=dim, correction=correction, keepdim=keepdim)

    std_dev = gems_sqrt(variance)

    return std_dev
