# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import threading

import torch

from flag_gems.ops.bmm_w8a8 import bmm_w8a8 as triton_bmm_w8a8
from flag_gems.ops.bmm_w8a8 import validate_inputs


class NativeState(threading.local):
    def __init__(self):
        self.operators = {}


_native_state = NativeState()


def native_gemm():
    device = torch.cuda.current_device()
    if device not in _native_state.operators:
        try:
            from mctlassEx import ScaledGEMM
        except ModuleNotFoundError as error:
            if error.name != "mctlassEx":
                raise
            _native_state.operators[device] = None
        else:
            _native_state.operators[device] = ScaledGEMM()
    return _native_state.operators[device]


def bmm_w8a8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    block_size=(128, 128, 128),
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
    *,
    config=None,
) -> torch.Tensor:
    """Channel-wise batched W8A8 matrix multiplication.

    A has shape (batch, M, K), B has shape (batch, K, N). A_scale is
    (batch, M) or (batch, M, 1); B_scale is (batch, N) or (batch, 1, N).
    INT8 accumulates in INT32; FP8 is converted to FP16 and accumulated
    in FP32. Scales are applied in FP32 before casting to out_dtype.
    block_size is retained for API compatibility and is unused here.
    """
    out = validate_inputs(A, B, A_scale, B_scale, out_dtype, out)
    batch, M, K = A.shape
    N = B.shape[2]
    if batch * M * N == 0:
        return out
    if (
        config is None
        and A.dtype == torch.int8
        and out_dtype == torch.bfloat16
        and K > 0
        and K % 32 == 0
        and N % 16 == 0
        and A.is_contiguous()
        and B.stride() == (K * N, 1, K)
        and out.is_contiguous()
        and A_scale.dtype == torch.float32
        and B_scale.dtype == torch.float32
        and A_scale.is_contiguous()
        and B_scale.is_contiguous()
    ):
        with torch.cuda.device(A.device):
            operator = native_gemm()
            if operator is None:
                return triton_bmm_w8a8(
                    A, B, A_scale, B_scale, block_size, out_dtype, out, config=config
                )
            operator(
                [M, N, K],
                A,
                B,
                out,
                A_scale.view(batch, M, 1),
                B_scale.view(batch, 1, N),
                None,
            )
        return out
    return triton_bmm_w8a8(
        A, B, A_scale, B_scale, block_size, out_dtype, out, config=config
    )
