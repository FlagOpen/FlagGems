import logging

import torch

logger = logging.getLogger(__name__)


def resolve_conj(A: torch.Tensor):
    logger.debug("GEMS RESOLVE_CONJ")
    return torch.complex(A.real, A.imag.neg()) if A.is_conj() else A
