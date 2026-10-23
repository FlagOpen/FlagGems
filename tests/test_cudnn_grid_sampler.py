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

# (N, C, H_in, W_in, H_out, W_out) tuples for the 2D cuDNN grid sampler.
if cfg.QUICK_MODE:
    # Single minimal case for --quick CI runs.
    GRID_SAMPLER_SHAPES = [
        (1, 3, 8, 8, 8, 8),
    ]
else:
    # Includes degenerate 1x1 inputs/grids, odd and non-square sizes, and
    # upsampled outputs.
    GRID_SAMPLER_SHAPES = [
        (1, 1, 1, 1, 1, 1),
        (1, 1, 2, 2, 4, 3),
        (1, 3, 8, 8, 8, 8),
        (2, 3, 16, 24, 16, 24),
        (2, 8, 32, 32, 32, 32),
        (4, 3, 17, 23, 33, 41),
        (2, 16, 64, 64, 128, 128),
        (1, 3, 5, 7, 1, 1),
    ]

# cudnn does not support bfloat16 for this operator (CUDNN_STATUS_BAD_PARAM);
# float16 and float32 are supported (cudnn computes in float32 internally).
GRID_SAMPLER_DTYPES = [torch.float16, torch.float32]


def _make_grid(N, H_out, W_out, dtype, device, mode="in_range"):
    """Build a grid tensor for the grid sampler.

    ``in_range`` samples the interior, ``full`` covers the whole [-1, 1] range
    (touching pixel centers), and ``out_of_range`` pushes part of the samples
    outside the input image so the zero-padding path is exercised.
    """
    grid = torch.randn(N, H_out, W_out, 2, dtype=torch.float32, device=device)
    if mode == "in_range":
        grid = torch.clamp(grid, -0.9, 0.9)
    elif mode == "full":
        grid = grid * 0.5
        grid = torch.clamp(grid, -1.0, 1.0)
    else:
        grid = grid * 1.5
    return grid.to(dtype)


def _make_inputs(N, C, H_in, W_in, H_out, W_out, dtype, mode):
    inp = torch.randn(N, C, H_in, W_in, dtype=dtype, device=flag_gems.device)
    grid = _make_grid(N, H_out, W_out, dtype, flag_gems.device, mode)
    # to_reference upcasts to float64, matching the reference out-buffer dtype.
    ref_inp = utils.to_reference(inp, True)
    ref_grid = utils.to_reference(grid, True)
    return inp, grid, ref_inp, ref_grid


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("shape", GRID_SAMPLER_SHAPES)
@pytest.mark.parametrize("dtype", GRID_SAMPLER_DTYPES)
@pytest.mark.parametrize("grid_mode", ["in_range", "full", "out_of_range"])
def test_cudnn_grid_sampler(shape, dtype, grid_mode):
    if TO_CPU:
        pytest.skip("cudnn_grid_sampler is CUDA-only, cannot run in quick-cpu mode")

    N, C, H_in, W_in, H_out, W_out = shape
    inp, grid, ref_inp, ref_grid = _make_inputs(
        N, C, H_in, W_in, H_out, W_out, dtype, grid_mode
    )

    ref_out = torch.ops.aten.cudnn_grid_sampler(ref_inp, ref_grid)
    res_out = flag_gems.cudnn_grid_sampler(inp, grid)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("dtype", GRID_SAMPLER_DTYPES)
def test_cudnn_grid_sampler_non_contiguous(dtype):
    if TO_CPU:
        pytest.skip("cudnn_grid_sampler is CUDA-only, cannot run in quick-cpu mode")

    N, C, H_in, W_in, H_out, W_out = 2, 4, 12, 16, 20, 24
    inp_base = torch.randn(N, C, W_in, H_in, dtype=dtype, device=flag_gems.device)
    grid_base = torch.randn(N, W_out, H_out, 2, dtype=dtype, device=flag_gems.device)
    inp = inp_base.transpose(2, 3)
    grid = grid_base.transpose(1, 2).clamp(-0.9, 0.9)

    ref_inp = utils.to_reference(inp, True)
    ref_grid = utils.to_reference(grid, True)
    ref_out = torch.ops.aten.cudnn_grid_sampler(ref_inp, ref_grid)
    res_out = flag_gems.cudnn_grid_sampler(inp, grid)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("dtype", GRID_SAMPLER_DTYPES)
def test_cudnn_grid_sampler_nan_inf_grid(dtype):
    if TO_CPU:
        pytest.skip("cudnn_grid_sampler is CUDA-only, cannot run in quick-cpu mode")

    N, C, H_in, W_in, H_out, W_out = 2, 3, 8, 8, 5, 6
    inp = torch.randn(N, C, H_in, W_in, dtype=dtype, device=flag_gems.device)
    grid = _make_grid(N, H_out, W_out, dtype, flag_gems.device, "in_range")
    grid[0, 0, 0, 0] = float("nan")
    grid[0, 1, 1, 1] = float("inf")
    grid[0, 2, 2, 0] = float("-inf")

    ref_inp = utils.to_reference(inp, True)
    ref_grid = utils.to_reference(grid, True)
    ref_out = torch.ops.aten.cudnn_grid_sampler(ref_inp, ref_grid)
    res_out = flag_gems.cudnn_grid_sampler(inp, grid)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.cudnn_grid_sampler
@pytest.mark.parametrize("dtype", GRID_SAMPLER_DTYPES)
def test_cudnn_grid_sampler_special_values(dtype):
    if TO_CPU:
        pytest.skip("cudnn_grid_sampler is CUDA-only, cannot run in quick-cpu mode")

    # Constant input: the interpolated value must equal that constant inside the
    # image and zero outside of it.
    N, C, H_in, W_in, H_out, W_out = 2, 3, 7, 9, 11, 13
    inp = torch.full((N, C, H_in, W_in), 2.5, dtype=dtype, device=flag_gems.device)
    grid = _make_grid(N, H_out, W_out, dtype, flag_gems.device, "full")

    ref_inp = utils.to_reference(inp, True)
    ref_grid = utils.to_reference(grid, True)
    ref_out = torch.ops.aten.cudnn_grid_sampler(ref_inp, ref_grid)
    res_out = flag_gems.cudnn_grid_sampler(inp, grid)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.cudnn_grid_sampler_out
@pytest.mark.parametrize("shape", GRID_SAMPLER_SHAPES)
@pytest.mark.parametrize("dtype", GRID_SAMPLER_DTYPES)
def test_cudnn_grid_sampler_out(shape, dtype):
    if TO_CPU:
        pytest.skip("cudnn_grid_sampler is CUDA-only, cannot run in quick-cpu mode")

    N, C, H_in, W_in, H_out, W_out = shape
    inp, grid, ref_inp, ref_grid = _make_inputs(
        N, C, H_in, W_in, H_out, W_out, dtype, "out_of_range"
    )
    # to_reference upcasts to float64, so the reference out buffer must match it.
    ref_out = torch.empty(
        (N, C, H_out, W_out), dtype=ref_inp.dtype, device=ref_inp.device
    )
    out = torch.empty((N, C, H_out, W_out), dtype=dtype, device=flag_gems.device)

    torch.ops.aten.cudnn_grid_sampler.out(ref_inp, ref_grid, out=ref_out)
    res_out = flag_gems.cudnn_grid_sampler_out(inp, grid, out=out)

    assert res_out.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref_out, dtype, atol=1e-3)
