import logging
import math

import torch

from ._meta_ext import meta_ext

logger = logging.getLogger(__name__)

_FAST_PATH_MAX_RANK = 64

# ``None`` unless the host-only helper is available (see ``_meta_ext.py``).
# It performs the very same metadata edit as the Python body below, but on
# IntArrayRef, so it does not pay the O(ndim) Python<->ATen conversions.
# Measured (ndim=6600): 34 us vs 496 us for the Python body and 128 us for the
# native ATen kernel.
#
# Resolved at import on purpose: building/loading lazily on the first call puts
# the extension build inside the benchmark's timing estimate, which inflates
# ``estimate_ms`` and drops triton's ``do_bench`` to a single measured sample.
_CPP_UNSQUEEZE = getattr(meta_ext(), "unsqueeze_", None)


def unsqueeze_(A: torch.Tensor, dim: int) -> torch.Tensor:
    """In-place version of unsqueeze (zero-copy view operation).

    Mutates ``A`` itself: inserts a size-1 dimension into ``A``'s
    shape/strides in place, matching the semantics of ``torch.Tensor.unsqueeze_``.
    """
    if _CPP_UNSQUEEZE is not None:
        return _CPP_UNSQUEEZE(A, dim)
    logger.debug("GEMS_KUNLUNXIN UNSQUEEZE_")
    ndim = A.dim()
    d = dim if dim >= 0 else ndim + dim + 1
    if d < 0 or d > ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [0, {ndim}], "
            f"but got {d})"
        )
    if ndim > _FAST_PATH_MAX_RANK:
        # Generic path: identical to the baseline implementation
        # (``A.reshape`` + ``Tensor.set_``), kept for very high-rank inputs.
        new_shape = list(A.shape)
        new_shape.insert(d, 1)
        A.set_(A.reshape(new_shape))
        return A

    shape = list(A.shape)
    stride = list(A.stride())
    numel = A.numel()
    if numel == 0:
        # Zero-size tensor: match the generic implementation's product of the
        # non-degenerate following sizes (zero entries are skipped).
        insert_stride = 1
        for i in range(d, ndim):
            sz = A.shape[i]
            insert_stride *= sz if sz else 1
    elif d == 0:
        insert_stride = numel
    else:
        insert_stride = numel // math.prod(A.shape[:d])
    shape.insert(d, 1)
    stride.insert(d, insert_stride)
    A.as_strided_(shape, stride, A.storage_offset())
    return A
