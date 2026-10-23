# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg
from .conftest import TO_CPU

# Custom shapes: (N, H, W, C); small edge cases plus odd/prime H, W values.
if cfg.QUICK_MODE:
    # Single minimal case for --quick CI runs.
    CUDNN_AFFINE_GRID_SHAPES = [
        (1, 2, 2, 3),
    ]
else:
    # (N, H, W, C) with degenerate, odd and non-square H, W combinations.
    CUDNN_AFFINE_GRID_SHAPES = [
        (1, 2, 2, 3),
        (1, 1, 5, 2),
        (2, 3, 1, 4),
        (2, 8, 8, 3),
        (4, 16, 16, 1),
        (3, 7, 11, 5),
        (2, 32, 32, 3),
    ]

# cuDNN computes this operator in float32 internally and rounds back to the
# input dtype, so bfloat16 is rejected by the reference kernel; float16,
# float32 and float64 are all supported.
CUDNN_AFFINE_GRID_FLOAT_DTYPES = [
    torch.float16,
    torch.float32,
    torch.float64,
]


def _reference(theta, N, C, H, W):
    return torch.cudnn_affine_grid_generator(theta, N, C, H, W)


@pytest.mark.cudnn_affine_grid_generator
@pytest.mark.parametrize("shape", CUDNN_AFFINE_GRID_SHAPES)
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_FLOAT_DTYPES)
def test_cudnn_affine_grid_generator(shape, dtype):
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator is CUDA-only, cannot run in quick-cpu mode"
        )

    N, H, W, C = shape
    theta = torch.randn((N, 2, 3), dtype=dtype, device=flag_gems.device)
    ref_theta = utils.to_reference(theta, True)

    ref_out = _reference(ref_theta, N, C, H, W)
    res_out = flag_gems.cudnn_affine_grid_generator(theta, N, C, H, W)

    assert res_out.shape == (N, H, W, 2)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cudnn_affine_grid_generator
@pytest.mark.parametrize("shape", CUDNN_AFFINE_GRID_SHAPES)
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_FLOAT_DTYPES)
def test_cudnn_affine_grid_generator_identity(shape, dtype):
    """With theta=identity the grid is the normalized coordinate lattice.

    cuDNN normalizes in the align_corners=True sense
    (``2 * i / (size - 1) - 1``) and collapses size-1 dimensions to -1.0.
    """
    N, H, W, C = shape
    theta = torch.zeros((N, 2, 3), dtype=dtype, device=flag_gems.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0

    # The identity reference is built with plain torch ops, so it runs wherever
    # the reference tensor lives (CPU in quick-cpu mode, GPU otherwise).
    ref_device = utils.to_reference(theta).device

    def coords(size):
        # cuDNN does the arithmetic in float32 and rounds to the output dtype.
        if size == 1:
            return torch.full((1,), -1.0, dtype=torch.float32, device=ref_device)
        idx = torch.arange(size, dtype=torch.float32, device=ref_device)
        return idx * (2.0 / (size - 1)) - 1.0

    norm_x = coords(W)
    norm_y = coords(H)
    ref_out = torch.stack(
        [
            norm_x[None, None, :].expand(N, H, W),
            norm_y[None, :, None].expand(N, H, W),
        ],
        dim=-1,
    ).to(dtype)

    res_out = flag_gems.cudnn_affine_grid_generator(theta, N, C, H, W)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cudnn_affine_grid_generator_out
@pytest.mark.parametrize("shape", CUDNN_AFFINE_GRID_SHAPES)
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_FLOAT_DTYPES)
def test_cudnn_affine_grid_generator_out(shape, dtype):
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator is CUDA-only, cannot run in quick-cpu mode"
        )

    N, H, W, C = shape
    theta = torch.randn((N, 2, 3), dtype=dtype, device=flag_gems.device)
    ref_theta = utils.to_reference(theta, True)
    # to_reference upcasts to float64, so the reference out buffer must match it.
    ref_out = torch.empty((N, H, W, 2), dtype=ref_theta.dtype, device=ref_theta.device)
    res_out = torch.empty((N, H, W, 2), dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.cudnn_affine_grid_generator.out(
        ref_theta, N, C, H, W, out=ref_out
    )
    res_out = flag_gems.cudnn_affine_grid_generator_out(theta, N, C, H, W, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cudnn_affine_grid_generator
@pytest.mark.parametrize("dtype", CUDNN_AFFINE_GRID_FLOAT_DTYPES)
def test_cudnn_affine_grid_generator_non_contiguous(dtype):
    """theta with non-contiguous strides must produce the same grid."""
    if TO_CPU:
        pytest.skip(
            "cudnn_affine_grid_generator is CUDA-only, cannot run in quick-cpu mode"
        )

    # Fixed shape (N, C, H, W) so theta can be made non-contiguous.
    N, C, H, W = 3, 4, 9, 13
    theta = torch.randn((N, 3, 2), dtype=dtype, device=flag_gems.device).transpose(1, 2)
    assert not theta.is_contiguous()

    ref_theta = utils.to_reference(theta, True)
    ref_out = _reference(ref_theta, N, C, H, W)
    res_out = flag_gems.cudnn_affine_grid_generator(theta, N, C, H, W)

    utils.gems_assert_close(res_out, ref_out, dtype)
