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


ORGQR_SHAPES = [(3, 0), (4, 3), (8, 5), (16, 8), (32, 16), (64, 32), (128, 64)]
ORGQR_BATCH_SHAPES = [(0, 4, 3), (2, 8, 5), (2, 3, 16, 8)]
ORGQR_DTYPES = [torch.float32, torch.float64]


def _make_reflectors(shape, dtype):
    torch.manual_seed(0)
    matrix = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return torch.geqrf(matrix)


@pytest.mark.orgqr
@pytest.mark.parametrize("shape", ORGQR_SHAPES + ORGQR_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", ORGQR_DTYPES)
def test_orgqr(shape, dtype):
    inp, tau = _make_reflectors(shape, dtype)
    ref = torch.orgqr(utils.to_reference(inp, True), utils.to_reference(tau, True))
    result = flag_gems.orgqr(inp, tau)

    assert result.shape == ref.shape
    assert result.dtype == ref.dtype
    utils.gems_assert_close(result, ref, dtype)


@pytest.mark.orgqr
@pytest.mark.parametrize("k", [0, 2, 5])
def test_orgqr_reflector_count(k):
    inp, tau = _make_reflectors((7, 5), torch.float32)
    tau = tau[:k]
    ref = torch.orgqr(utils.to_reference(inp, True), utils.to_reference(tau, True))
    result = flag_gems.orgqr(inp, tau)
    utils.gems_assert_close(result, ref, torch.float32)


@pytest.mark.orgqr
@pytest.mark.parametrize(
    "input_shape,tau_shape", [((2, 3), (2,)), ((4, 3), (4,))]
)
def test_orgqr_rejects_invalid_shapes(input_shape, tau_shape):
    torch.manual_seed(0)
    with pytest.raises(RuntimeError):
        flag_gems.orgqr(
            torch.randn(input_shape, device=flag_gems.device),
            torch.randn(tau_shape, device=flag_gems.device),
        )


@pytest.mark.orgqr
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_orgqr_complex_is_explicitly_unsupported(dtype):
    inp, tau = _make_reflectors((4, 3), dtype)
    assert torch.orgqr(inp, tau).dtype == dtype
    with pytest.raises(RuntimeError, match="complex dtypes are not supported"):
        flag_gems.orgqr(inp, tau)


# orgqr.out is registered by the same family but is not a separate spreadsheet ID.
# Its overload contract remains covered in this family correctness artifact.
@pytest.mark.orgqr_out
@pytest.mark.parametrize("shape", ORGQR_SHAPES)
@pytest.mark.parametrize("dtype", ORGQR_DTYPES)
def test_orgqr_out(shape, dtype):
    inp, tau = _make_reflectors(shape, dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_tau = utils.to_reference(tau, True)
    ref_out = torch.empty(0, dtype=dtype, device=ref_inp.device)
    ref = torch.orgqr(ref_inp, ref_tau, out=ref_out)
    out = torch.empty(0, dtype=dtype, device=flag_gems.device)
    result = flag_gems.orgqr_out(inp, tau, out=out)

    assert result is out
    assert ref is ref_out
    assert out.shape == inp.shape
    utils.gems_assert_close(out, ref, dtype)


@pytest.mark.orgqr_out
@pytest.mark.parametrize("dtype", ORGQR_DTYPES)
@pytest.mark.parametrize("layout", ["transposed_batch", "sliced_batch"])
def test_orgqr_out_noncontiguous_batch_dimensions(dtype, layout):
    inp, tau = _make_reflectors((2, 3, 7, 5), dtype)
    ref_input = utils.to_reference(inp, True)
    ref_tau = utils.to_reference(tau, True)

    def make_out(device):
        if layout == "transposed_batch":
            return torch.empty((3, 2, 7, 5), dtype=dtype, device=device).transpose(
                0, 1
            )
        return torch.empty((4, 3, 7, 5), dtype=dtype, device=device)[::2]

    out = make_out(flag_gems.device)
    ref_out = make_out(ref_input.device)
    original_shape = out.shape
    original_stride = out.stride()
    original_pointer = out.data_ptr()
    assert not out.is_contiguous()
    with pytest.raises(RuntimeError):
        out.view(6, 7, 5)

    reference = torch.orgqr(ref_input, ref_tau, out=ref_out)
    result = flag_gems.orgqr_out(inp, tau, out=out)

    assert result is out
    assert out.shape == original_shape
    assert out.stride() == original_stride
    assert out.data_ptr() == original_pointer
    utils.gems_assert_close(out, reference, dtype)


@pytest.mark.orgqr_out
@pytest.mark.skipif(not utils.fp64_is_supported, reason="FP64 is not supported")
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("nonflattenable", [False, True])
def test_orgqr_out_casts_only_final_result(output_dtype, nonflattenable):
    torch.manual_seed(0)
    matrix = torch.randn((2, 3, 31, 19), dtype=torch.float64, device=flag_gems.device)
    inp, tau = torch.geqrf(matrix)
    ref_input, ref_tau = utils.to_reference(inp, True), utils.to_reference(tau, True)
    if nonflattenable:
        out = torch.empty(
            (3, 2, 31, 19), dtype=output_dtype, device=inp.device
        ).transpose(0, 1)
        ref_out = torch.empty(
            (3, 2, 31, 19), dtype=output_dtype, device=ref_input.device
        ).transpose(0, 1)
    else:
        out = torch.empty(matrix.shape, dtype=output_dtype, device=inp.device)
        ref_out = torch.empty(matrix.shape, dtype=output_dtype, device=ref_input.device)
    original_pointer, original_stride = out.data_ptr(), out.stride()
    reference = torch.orgqr(ref_input, ref_tau, out=ref_out)
    result = flag_gems.orgqr_out(inp, tau, out=out)

    assert result is out and result.data_ptr() == original_pointer
    assert result.stride() == original_stride
    actual, expected = utils.to_cpu(result, reference), reference
    precision = torch.finfo(output_dtype)
    torch.testing.assert_close(
        actual, expected, rtol=precision.eps, atol=precision.tiny * precision.eps
    )


@pytest.mark.orgqr_out
def test_orgqr_out_stride_alias_dtype_and_device_contract():
    inp, tau = _make_reflectors((7, 5), torch.float32)
    ref_input = utils.to_reference(inp, True)
    ref_tau = utils.to_reference(tau, True)

    out = torch.empty((5, 7), device=flag_gems.device).T
    ref_out = torch.empty((5, 7), device=ref_input.device).T
    result = flag_gems.orgqr_out(inp, tau, out=out)
    ref = torch.orgqr(ref_input, ref_tau, out=ref_out)
    assert result is out
    assert ref is ref_out
    assert not result.is_contiguous()
    utils.gems_assert_close(result, ref, torch.float32)

    alias_input, alias_tau = _make_reflectors((7, 5), torch.float32)
    ref_alias_input = utils.to_reference(alias_input, True).clone()
    ref_alias_tau = utils.to_reference(alias_tau, True)
    ref_alias = torch.orgqr(ref_alias_input, ref_alias_tau, out=ref_alias_input)
    result = flag_gems.orgqr_out(alias_input, alias_tau, out=alias_input)
    assert result is alias_input
    utils.gems_assert_close(result, ref_alias, torch.float32)

    wide_out = torch.empty((7, 5), dtype=torch.float64, device=flag_gems.device)
    wide_ref = torch.empty((7, 5), dtype=torch.float64, device=ref_input.device)
    result = flag_gems.orgqr_out(inp, tau, out=wide_out)
    ref = torch.orgqr(ref_input, ref_tau, out=wide_ref)
    utils.gems_assert_close(result, ref, torch.float64)

    bad_out = torch.empty((7, 5), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.orgqr(inp, tau, out=bad_out)
    with pytest.raises(RuntimeError):
        flag_gems.orgqr_out(inp, tau, out=bad_out)

    cpu_out = torch.empty((7, 5), device="cpu")
    with pytest.raises(RuntimeError):
        torch.orgqr(inp, tau, out=cpu_out)
    with pytest.raises(RuntimeError):
        flag_gems.orgqr_out(inp, tau, out=cpu_out)
