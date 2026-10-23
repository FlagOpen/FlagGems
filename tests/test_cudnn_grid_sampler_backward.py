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
"""Accuracy tests for ``cudnn_grid_sampler_backward``.

``aten::cudnn_grid_sampler_backward`` is the backward pass of cuDNN's grid
sampler, i.e. ``grid_sampler_2d_backward`` with bilinear interpolation, zero
padding and ``align_corners=True``.  The reference below uses that public ATen
op so it also runs on CPU in the ``--ref=cpu --quick`` CI mode.  cuDNN does not
support ``bfloat16`` for this operator, so bfloat16 inputs are only exercised
through the Triton path against a float64 reference.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# cuDNN accepts float16/float32/float64 only; bfloat16 still goes through the
# Triton kernel but must be compared against a float32 reference.
GRID_SAMPLER_BACKWARD_DTYPES = [torch.float16, torch.float32, torch.bfloat16]

# (N, C, H_in, W_in, H_out, W_out) tuples for the 2D cuDNN grid sampler
# backward.  Includes degenerate 1x1 inputs, odd and non-square sizes, and
# downsampled/upsampled outputs.
SHAPES = [
    (1, 1, 4, 4, 2, 2),
    (2, 3, 8, 8, 8, 8),
    (2, 3, 9, 7, 5, 4),
    (4, 8, 16, 16, 17, 13),
    (1, 32, 32, 24, 30, 32),
    (2, 16, 56, 56, 28, 28),
    (1, 3, 1, 1, 4, 5),
    (3, 5, 2, 7, 1, 1),
]

# Grid coordinates that clearly fall outside of the input image.
OUT_OF_BOUND_SHAPES = [
    (2, 3, 10, 10, 6, 6),
    (1, 4, 3, 3, 5, 5),
]


def _make_inputs(shape, dtype, grid_range, seed=None):
    N, C, H_in, W_in, H_out, W_out = shape
    if seed is not None:
        torch.manual_seed(seed)
    inp = torch.randn((N, C, H_in, W_in), dtype=dtype, device=flag_gems.device)
    grid = torch.empty(
        (N, H_out, W_out, 2), dtype=dtype, device=flag_gems.device
    ).uniform_(*grid_range)
    grad_output = torch.randn(
        (N, C, H_out, W_out), dtype=dtype, device=flag_gems.device
    )
    return inp, grid, grad_output


def _reference_grads(inp, grid, grad_output, dtype):
    """Reference backward pass computed with the public ATen op in float64.

    cudnn_grid_sampler_backward is grid_sampler_2d_backward with bilinear
    interpolation, zero padding and align_corners=True; the ATen op exists on
    CPU too, so the reference also works in the quick-cpu CI mode.
    """
    ref_grad_output = utils.to_reference(grad_output, True)
    ref_inp = utils.to_reference(inp, True)
    ref_grid = utils.to_reference(grid, True)
    ref_grad_input, ref_grad_grid = torch.ops.aten.grid_sampler_2d_backward(
        ref_grad_output, ref_inp, ref_grid, 0, 0, True, [True, True]
    )
    return ref_grad_input.to(dtype), ref_grad_grid.to(dtype)


def _assert_grads(
    res_grad_input, res_grad_grid, ref_grad_input, ref_grad_grid, dtype, C
):
    """grad_input is a pure scatter, grad_grid reduces over the channels."""
    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
    # The grid gradient reduces over C channels, so fp32 rounding accumulates
    # a little more than the default 1e-4 tolerance on large-channel shapes.
    utils.gems_assert_close(
        res_grad_grid, ref_grad_grid, dtype, reduce_dim=C, atol=2e-3
    )


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward(shape, dtype):
    inp, grid, grad_output = _make_inputs(shape, dtype, (-1.0, 1.0))

    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    _assert_grads(
        res_grad_input, res_grad_grid, ref_grad_input, ref_grad_grid, dtype, shape[1]
    )


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize("shape", OUT_OF_BOUND_SHAPES)
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward_out_of_bound(shape, dtype):
    # Grid values in [-2, 2] exercise zero padding in every direction.
    inp, grid, grad_output = _make_inputs(shape, dtype, (-2.0, 2.0))

    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    _assert_grads(
        res_grad_input, res_grad_grid, ref_grad_input, ref_grad_grid, dtype, shape[1]
    )


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward_edge_coordinates(dtype):
    # Coordinates sitting exactly on the corners of the image, plus values just
    # inside the border, must reproduce cuDNN's convention.
    shape = (2, 3, 6, 5, 4, 4)
    inp, _, grad_output = _make_inputs(shape, dtype, (-1.0, 1.0), seed=0)

    corners = torch.tensor(
        [
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-0.999, 0.999],
            [0.0, 0.0],
            [-0.5, 0.25],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-0.75, -0.25],
            [0.75, 0.25],
            [-0.25, -0.75],
            [0.25, 0.75],
            [-0.001, 0.001],
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    grid = corners.reshape(1, 4, 4, 2).expand(shape[0], 4, 4, 2).contiguous()

    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    _assert_grads(
        res_grad_input, res_grad_grid, ref_grad_input, ref_grad_grid, dtype, shape[1]
    )


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward_all_out_of_bound(dtype):
    # Every sample point is outside of the image => zero gradients everywhere.
    shape = (2, 3, 5, 5, 3, 3)
    inp, _, _ = _make_inputs(shape, dtype, (-1.0, 1.0), seed=1)
    grad_output = torch.randn(
        (shape[0], shape[1], shape[4], shape[5]), dtype=dtype, device=flag_gems.device
    )
    grid = torch.full(
        (shape[0], shape[4], shape[5], 2), 3.0, dtype=dtype, device=flag_gems.device
    )

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    # to_reference moves the expected-zero buffers to the reference device in
    # the quick-cpu CI mode.
    zero_input = utils.to_reference(torch.zeros_like(res_grad_input))
    zero_grid = utils.to_reference(torch.zeros_like(res_grad_grid))
    utils.gems_assert_close(res_grad_input, zero_input, dtype, atol=0.0)
    utils.gems_assert_close(res_grad_grid, zero_grid, dtype, atol=0.0)


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward_accumulation(dtype):
    # A single sample point only scatters into four input pixels; all other
    # pixels must stay exactly zero.
    shape = (1, 2, 8, 8, 1, 1)
    inp, _, grad_output = _make_inputs(shape, dtype, (-0.5, 0.5), seed=2)
    grid = torch.zeros((1, 1, 1, 2), dtype=dtype, device=flag_gems.device)

    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    _assert_grads(
        res_grad_input, res_grad_grid, ref_grad_input, ref_grad_grid, dtype, shape[1]
    )


@pytest.mark.cudnn_grid_sampler_backward
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 4, 4, 1, 1),
        (2, 3, 8, 8, 8, 8),
        (1, 3, 1, 1, 4, 5),
    ],
)
def test_cudnn_grid_sampler_backward_float64(shape):
    dtype = torch.float64
    inp, grid, grad_output = _make_inputs(shape, dtype, (-1.5, 1.5), seed=3)

    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    res_grad_input, res_grad_grid = flag_gems.cudnn_grid_sampler_backward(
        inp, grid, grad_output
    )

    _assert_grads(
        res_grad_input, res_grad_grid, ref_grad_input, ref_grad_grid, dtype, shape[1]
    )


@pytest.mark.cudnn_grid_sampler_backward_out
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", GRID_SAMPLER_BACKWARD_DTYPES)
def test_cudnn_grid_sampler_backward_out(shape, dtype):
    N, C, _, _, H_out, W_out = shape
    inp, grid, grad_output = _make_inputs(shape, dtype, (-1.0, 1.0))

    ref_grad_input, ref_grad_grid = _reference_grads(inp, grid, grad_output, dtype)

    out0 = torch.empty_like(inp)
    out1 = torch.empty((N, H_out, W_out, 2), dtype=dtype, device=flag_gems.device)

    res_out0, res_out1 = flag_gems.cudnn_grid_sampler_backward_out(
        inp, grid, grad_output, out0, out1
    )

    assert res_out0.data_ptr() == out0.data_ptr()
    assert res_out1.data_ptr() == out1.data_ptr()
    _assert_grads(res_out0, res_out1, ref_grad_input, ref_grad_grid, dtype, C)
