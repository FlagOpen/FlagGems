# Copyright 2026 FlagOS Contributors
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

from .conftest import TO_CPU

# Shapes representative of common batch-norm workloads.
# The feature dimension is always shape[1].
SHAPES = [
    (16, 3, 32),
    (32, 32, 32),
    (8, 32, 224, 224),
    (32, 64, 56, 56),
    (64, 128, 28, 28),
    (16, 256, 14, 14),
]


@pytest.mark.batch_norm_gather_stats_with_counts
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("num_segments", [2, 4])
def test_batch_norm_gather_stats_with_counts(shape, dtype, num_segments):
    # batch_norm_gather_stats_with_counts has no CPU backend in PyTorch
    torch.manual_seed(0)
    if TO_CPU:
        pytest.skip(
            "batch_norm_gather_stats_with_counts has no CPU backend; skip under --ref=cpu"
        )

    C = shape[1]
    N = shape[0]
    spatial = 1
    for s in shape[2:]:
        spatial *= s
    count_per_segment = N * spatial

    input_t = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mean = torch.randn(num_segments, C, dtype=torch.float32, device=flag_gems.device)
    invstd = (
        torch.rand(num_segments, C, dtype=torch.float32, device=flag_gems.device) + 0.1
    )
    running_mean = torch.randn(C, dtype=torch.float32, device=flag_gems.device)
    running_var = torch.rand(C, dtype=torch.float32, device=flag_gems.device) + 0.1
    ref_input = utils.to_reference(input_t, True).to(dtype)
    ref_mean = utils.to_reference(mean, False)
    ref_invstd = utils.to_reference(invstd, False)
    ref_running_mean = utils.to_reference(running_mean, False)
    ref_running_var = utils.to_reference(running_var, False)
    # Per-segment counts tensor (float)
    counts = torch.full(
        (num_segments,), count_per_segment, dtype=torch.float32, device=flag_gems.device
    )

    ref_running_mean_arg = ref_running_mean.clone()
    ref_running_var_arg = ref_running_var.clone()
    actual_running_mean = running_mean.clone()
    actual_running_var = running_var.clone()
    mean_pointer, var_pointer = (
        actual_running_mean.data_ptr(),
        actual_running_var.data_ptr(),
    )
    mean_stride, var_stride = actual_running_mean.stride(), actual_running_var.stride()
    ref_out = torch.batch_norm_gather_stats_with_counts(
        ref_input,
        ref_mean,
        ref_invstd,
        ref_running_mean_arg,
        ref_running_var_arg,
        0.1,
        1e-5,
        utils.to_reference(counts, False),
    )

    res_out = flag_gems.batch_norm_gather_stats_with_counts(
        input_t,
        mean,
        invstd,
        actual_running_mean,
        actual_running_var,
        0.1,
        1e-5,
        counts,
    )

    for ref_val, res_val in zip(ref_out, res_out):
        # Output is always float32 regardless of input dtype
        utils.gems_assert_close(res_val, ref_val, torch.float32)
    assert actual_running_mean.data_ptr() == mean_pointer
    assert actual_running_var.data_ptr() == var_pointer
    assert actual_running_mean.stride() == mean_stride
    assert actual_running_var.stride() == var_stride
    utils.gems_assert_close(actual_running_mean, ref_running_mean_arg, torch.float32)
    utils.gems_assert_close(actual_running_var, ref_running_var_arg, torch.float32)
