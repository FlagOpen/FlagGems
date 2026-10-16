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


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


_POINT_CASES = [
    pytest.param((50, 2), [4, 5], id="2d"),
    pytest.param((100, 3), [3, 4, 5], id="3d"),
    pytest.param((80, 4), [2, 3, 4, 5], id="4d"),
]


def _reference(inp, bins, *, range=None, weight=None, density=False):
    ref_inp = utils.to_reference(inp, True).to("cpu")
    if isinstance(bins, (tuple, list)) and bins and isinstance(bins[0], torch.Tensor):
        ref_bins = tuple(utils.to_reference(item, True).to("cpu") for item in bins)
    else:
        ref_bins = bins
    ref_weight = (
        utils.to_reference(weight, True).to("cpu") if weight is not None else None
    )
    hist, edges = torch.histogramdd(
        ref_inp, bins=ref_bins, range=range, weight=ref_weight, density=density
    )
    return hist.to(inp.dtype), tuple(edge.to(inp.dtype) for edge in edges)


def _assert_result(result, reference, dtype):
    hist, edges = result
    ref_hist, ref_edges = reference
    if dtype in utils.ALL_FLOAT_DTYPES:
        utils.gems_assert_close(hist.cpu(), ref_hist, dtype, atol=1e-4)
    else:
        utils.gems_assert_equal(hist.cpu(), ref_hist)
    assert hist.shape == ref_hist.shape
    assert hist.dtype == ref_hist.dtype
    assert len(edges) == len(ref_edges)
    for edge, ref_edge in zip(edges, ref_edges):
        utils.gems_assert_close(edge.cpu(), ref_edge, dtype, atol=1e-4)
        assert edge.shape == ref_edge.shape
        assert edge.dtype == ref_edge.dtype


@pytest.mark.histogramdd
@pytest.mark.parametrize("shape,bins", _POINT_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_histogramdd_bin_counts(shape, bins, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    before = inp.clone()
    reference = _reference(inp, bins)

    with flag_gems.use_gems():
        result = torch.histogramdd(inp, bins=bins)

    _assert_result(result, reference, dtype)
    utils.gems_assert_equal(inp, before)


@pytest.mark.histogramdd
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("density", [False, True])
def test_histogramdd_range_weight_density(dtype, weighted, density):
    torch.manual_seed(0)
    inp = torch.randn((60, 2), dtype=dtype, device=flag_gems.device)
    weight = (
        torch.rand((60,), dtype=dtype, device=flag_gems.device)
        if weighted
        else None
    )
    range_ = [-2.0, 2.0, -3.0, 3.0]
    reference = _reference(
        inp, [5, 4], range=range_, weight=weight, density=density
    )

    with flag_gems.use_gems():
        result = torch.histogramdd(
            inp, bins=[5, 4], range=range_, weight=weight, density=density
        )

    _assert_result(result, reference, dtype)


@pytest.mark.histogramdd
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_histogramdd_tensor_list_bins(dtype):
    inp = torch.tensor(
        [[-1.0, 0.0], [0.25, 0.5], [1.0, 1.5], [2.0, -0.5]],
        dtype=dtype,
        device=flag_gems.device,
    )
    bins = (
        torch.tensor([-2.0, 0.0, 1.0, 3.0], dtype=dtype, device=flag_gems.device),
        torch.tensor([-1.0, 0.25, 1.0, 2.0], dtype=dtype, device=flag_gems.device),
    )
    reference = _reference(inp, bins)

    with flag_gems.use_gems():
        result = torch.histogramdd(inp, bins=bins)

    _assert_result(result, reference, dtype)


@pytest.mark.histogramdd
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_histogramdd_integer_bins_overload(dtype):
    torch.manual_seed(0)
    inp = torch.randn((40, 3), dtype=dtype, device=flag_gems.device)
    reference = _reference(inp, 4)

    with flag_gems.use_gems():
        result = torch.histogramdd(inp, bins=4)

    _assert_result(result, reference, dtype)


@pytest.mark.histogramdd
def test_histogramdd_empty_and_degenerate_range():
    empty = torch.empty((0, 2), dtype=torch.float32, device=flag_gems.device)
    with flag_gems.use_gems():
        empty_result = torch.histogramdd(empty, bins=[3, 3])
    _assert_result(empty_result, _reference(empty, [3, 3]), torch.float32)

    inp = torch.full((10, 2), 5.0, device=flag_gems.device)
    with flag_gems.use_gems():
        result = torch.histogramdd(inp, bins=[3, 3])
    _assert_result(result, _reference(inp, [3, 3]), torch.float32)


@pytest.mark.histogramdd
def test_histogramdd_noncontiguous_input():
    torch.manual_seed(0)
    inp = torch.randn((2, 32), device=flag_gems.device).transpose(0, 1)
    assert not inp.is_contiguous()
    reference = _reference(inp, [4, 5])

    with flag_gems.use_gems():
        result = torch.histogramdd(inp, bins=[4, 5])

    _assert_result(result, reference, torch.float32)


@pytest.mark.histogramdd
def test_histogramdd_rejects_mismatched_bins():
    inp = torch.randn((20, 2), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.histogramdd(inp, bins=[4, 5, 6])
