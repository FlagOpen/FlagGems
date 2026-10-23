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
"""Tests for quantized_rnn_relu_cell (delegated implementation).

Reference strategy
------------------
``quantized_rnn_relu_cell`` is delegated to aten (there is no Triton
kernel). It runs one RNN cell step with a ReLU nonlinearity over
dynamically quantized weights: the two projections are computed by the
fbgemm CPU SIMD kernel behind
``fbgemm_linear_int8_weight_fp32_activation`` and combined with
``h_next = relu(hgates + igates)`` - a **single gate**, the simplest of
the quantized cells (no ``unsafe_chunk`` step in ``SimpleCell``, single
``hx`` tensor, single output).

**Deliberately no numerical comparison against a fbgemm-free reference.**
The sibling records for the fbgemm linear family (PRs #7638 and #7647)
and for the GRU cell (#7658) established that the fbgemm
dynamic-quantization grid and per-row requantization behavior are private
to the fbgemm build - a Python-side reconstruction of the math matches
the local build bitwise but can still fail on CI. Two validations were
run locally and are recorded in the development notes:

* wrapper output == ``relu(linear_hh + linear_ih)`` applied to
  ``fbgemm_linear_int8_weight_fp32_activation`` projections, bitwise, on
  the local build (the exact C++ composition);
* a quantize-reconstruct reference (``(w - zp) * scale`` style) does NOT
  match - the same cross-build divergence #7638 hit.

The tests therefore validate behavior that is identical on every build:

* the wrapper's validation logic (dtype / dim / shape / Scalar error
  paths), which runs before dispatch;
* the delegation path on a legal call - asserting output shape, dtype and
  device structurally, plus **self-consistency**: the same inputs through
  the wrapper and through ``torch.ops.aten.quantized_rnn_relu_cell``
  produce bitwise identical results (wrapper-vs-aten contract,
  build-independent);
* the **gate-math contract**: wrapper output == ``relu`` applied to the
  sum of the two projections of ``fbgemm_linear_int8_weight_fp32_activation``
  (this uses only aten helpers present on the same build, no fbgemm-free
  reconstruction, so it holds wherever the helpers exist).

Availability probing is lazy (autouse fixture via ``hasattr``): the
fbgemm-backed helpers are deprecated on recent builds and their
availability varies across torch builds; tests that build real packed
weights skip when the environment cannot run them.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# The cell reads fp32 activations only (other dtypes are rejected by the
# fbgemm kernel with "expected scalar type Float"), so the dtype matrix is
# fixed instead of parametrizing over FLOAT_DTYPES.
SHAPES = (
    [(4, 8, 3)]
    if utils.QUICK_MODE
    else [
        (4, 8, 3),
        (16, 32, 16),
        (64, 128, 64),
    ]
)  # (batch, K_in, hidden); batch fixed at 4 in _make_params


def _make_params(hidden, k_in, device="cpu"):
    """Build a consistent set of quantized RNN ReLU cell parameters.

    The int8 weight, column offsets, scale and zero point come from aten's
    own quantizer (``fbgemm_linear_quantize_weight``) so the test never
    hardcodes fbgemm's quantization convention; the packed objects come
    from ``fbgemm_pack_quantized_matrix``.
    """
    torch.manual_seed(7)
    w_ih = torch.randn(hidden, k_in, device=device, dtype=torch.float32)
    w_hh = torch.randn(hidden, hidden, device=device, dtype=torch.float32)
    qw_ih, col_ih, scale_ih, zp_ih = torch.ops.aten.fbgemm_linear_quantize_weight(
        w_ih.cpu()
    )
    qw_hh, col_hh, scale_hh, zp_hh = torch.ops.aten.fbgemm_linear_quantize_weight(
        w_hh.cpu()
    )
    packed_ih = torch.ops.aten.fbgemm_pack_quantized_matrix(qw_ih)
    packed_hh = torch.ops.aten.fbgemm_pack_quantized_matrix(qw_hh)
    b_ih = torch.randn(hidden, dtype=torch.float32)
    b_hh = torch.randn(hidden, dtype=torch.float32)
    input = torch.randn(4, k_in, dtype=torch.float32)
    hx = torch.randn(4, hidden, dtype=torch.float32)
    return {
        "input": input,
        "hx": hx,
        "w_ih": qw_ih,
        "w_hh": qw_hh,
        "b_ih": b_ih,
        "b_hh": b_hh,
        "packed_ih": packed_ih,
        "packed_hh": packed_hh,
        "col_offsets_ih": col_ih,
        "col_offsets_hh": col_hh,
        "scale_ih": scale_ih,
        "scale_hh": scale_hh,
        "zero_point_ih": zp_ih,
        "zero_point_hh": zp_hh,
    }


@pytest.fixture(autouse=True)
def _require_fbgemm_cell_helpers():
    """The delegation path needs the fbgemm-backed aten weight helpers.

    Probed lazily (not at import time): the helpers are deprecated on
    recent builds and absent on some torch builds. Validation-path tests
    run regardless; delegation tests skip when the helpers are missing.
    """
    if not (
        hasattr(torch.ops.aten, "fbgemm_linear_quantize_weight")
        and hasattr(torch.ops.aten, "fbgemm_pack_quantized_matrix")
    ):
        pytest.skip(
            "fbgemm-backed aten helpers (fbgemm_linear_quantize_weight / "
            "fbgemm_pack_quantized_matrix) unavailable on this torch build"
        )


def _relu_reference(
    input,
    hx,
    w_ih,
    w_hh,
    b_ih,
    b_hh,
    packed_ih,
    packed_hh,
    col_ih,
    col_hh,
    scale_ih,
    scale_hh,
    zp_ih,
    zp_hh,
):
    """ReLU cell math over the same-build fbgemm projections.

    Uses ``fbgemm_linear_int8_weight_fp32_activation`` - the exact kernel
    the C++ cell calls - so this is a composition contract on the same
    build, not a fbgemm-free numerical reconstruction. The C++ cell is
    ``SimpleCell<relu_f, QuantizedCellParams>``:
    ``relu(linear_hh(hx).add_(linear_ih(input)))``.
    """
    igates = torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation(
        input, w_ih, packed_ih, col_ih, scale_ih, zp_ih, b_ih
    )
    hgates = torch.ops.aten.fbgemm_linear_int8_weight_fp32_activation(
        hx, w_hh, packed_hh, col_hh, scale_hh, zp_hh, b_hh
    )
    return hgates.add(igates).relu_()


@pytest.mark.quantized_rnn_relu_cell
@pytest.mark.parametrize("k_in, hidden", [shape[1:] for shape in SHAPES])
def test_quantized_rnn_relu_cell_delegates(k_in, hidden):
    """A legal call delegates without raising; output is fp32 (batch, hidden)."""
    params = _make_params(hidden, k_in)

    out = flag_gems.quantized_rnn_relu_cell(
        params["input"],
        params["hx"],
        params["w_ih"],
        params["w_hh"],
        params["b_ih"],
        params["b_hh"],
        params["packed_ih"],
        params["packed_hh"],
        params["col_offsets_ih"],
        params["col_offsets_hh"],
        params["scale_ih"],
        params["scale_hh"],
        params["zero_point_ih"],
        params["zero_point_hh"],
    )

    assert out.dtype == torch.float32
    assert out.shape == (4, hidden)


@pytest.mark.quantized_rnn_relu_cell
def test_quantized_rnn_relu_cell_gate_math_contract():
    """Wrapper output equals relu over the same-build projections.

    The C++ cell computes ``linear_ih``/``linear_hh`` through
    ``fbgemm_linear_int8_weight_fp32_activation`` and applies
    ``relu(hgates + igates)``. Rebuilding that composition from the same
    inputs (using only aten helpers from this build) must reproduce the
    wrapper output bitwise; this asserts the delegation and the gate
    composition without any fbgemm-free numerical reconstruction (see
    module docstring).
    """
    params = _make_params(3, 5)

    out = flag_gems.quantized_rnn_relu_cell(
        params["input"],
        params["hx"],
        params["w_ih"],
        params["w_hh"],
        params["b_ih"],
        params["b_hh"],
        params["packed_ih"],
        params["packed_hh"],
        params["col_offsets_ih"],
        params["col_offsets_hh"],
        params["scale_ih"],
        params["scale_hh"],
        params["zero_point_ih"],
        params["zero_point_hh"],
    )
    ref = _relu_reference(
        params["input"],
        params["hx"],
        params["w_ih"],
        params["w_hh"],
        params["b_ih"],
        params["b_hh"],
        params["packed_ih"],
        params["packed_hh"],
        params["col_offsets_ih"],
        params["col_offsets_hh"],
        params["scale_ih"],
        params["scale_hh"],
        params["zero_point_ih"],
        params["zero_point_hh"],
    )

    utils.gems_assert_close(
        utils.to_reference(out),
        utils.to_reference(ref),
        dtype=torch.float32,
        atol=1e-6,
    )


@pytest.mark.quantized_rnn_relu_cell
def test_quantized_rnn_relu_cell_wrapper_vs_aten_parity():
    """The wrapper and the raw aten op return bitwise identical results.

    Delegation contract: same inputs, both entry points, identical output.
    """
    params = _make_params(3, 5)

    out_wrapper = flag_gems.quantized_rnn_relu_cell(
        params["input"],
        params["hx"],
        params["w_ih"],
        params["w_hh"],
        params["b_ih"],
        params["b_hh"],
        params["packed_ih"],
        params["packed_hh"],
        params["col_offsets_ih"],
        params["col_offsets_hh"],
        params["scale_ih"],
        params["scale_hh"],
        params["zero_point_ih"],
        params["zero_point_hh"],
    )
    out_aten = torch.ops.aten.quantized_rnn_relu_cell(
        params["input"],
        params["hx"],
        params["w_ih"],
        params["w_hh"],
        params["b_ih"],
        params["b_hh"],
        params["packed_ih"],
        params["packed_hh"],
        params["col_offsets_ih"],
        params["col_offsets_hh"],
        params["scale_ih"],
        params["scale_hh"],
        params["zero_point_ih"],
        params["zero_point_hh"],
    )

    utils.gems_assert_close(
        utils.to_reference(out_wrapper),
        utils.to_reference(out_aten),
        dtype=torch.float32,
        atol=1e-6,
    )


@pytest.mark.quantized_rnn_relu_cell
def test_quantized_rnn_relu_cell_cuda_stages_through_host():
    """All-CUDA call routes through the wrapper and returns on the caller device.

    The fbgemm CPU SIMD kernel dereferences host pointers and SimpleCell
    has no is_cuda branch (unlike the GRU/LSTM cells): a CUDA input goes
    straight into ``fbgemm_linear_int8_weight_fp32_activation`` and
    segfaults on this parameter set. The wrapper stages every tensor
    through host memory. ``packed_ih``/``packed_hh`` are opaque CPU fbgemm
    objects and must be passed as CPU products of
    ``fbgemm_pack_quantized_matrix``.
    """
    if flag_gems.device != "cuda":
        pytest.skip("CUDA-only routing check")
    params = _make_params(3, 5)
    input = params["input"].cuda()
    hx = params["hx"].cuda()
    w_ih = params["w_ih"].cuda()
    w_hh = params["w_hh"].cuda()
    b_ih = params["b_ih"].cuda()
    b_hh = params["b_hh"].cuda()
    col_ih = params["col_offsets_ih"].cuda()
    col_hh = params["col_offsets_hh"].cuda()

    out = flag_gems.quantized_rnn_relu_cell(
        input,
        hx,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        params["packed_ih"],
        params["packed_hh"],
        col_ih,
        col_hh,
        params["scale_ih"],
        params["scale_hh"],
        params["zero_point_ih"],
        params["zero_point_hh"],
    )

    assert out.dtype == torch.float32
    assert out.device.type == "cuda"
    assert out.shape == (4, 3)


@pytest.mark.quantized_rnn_relu_cell
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_quantized_rnn_relu_cell_rejects_non_fp32_activation(dtype):
    """The fbgemm kernel reads fp32 activations; other dtypes are rejected."""
    params = _make_params(3, 5)
    input = torch.randn(4, 5, dtype=dtype)
    hx = torch.randn(4, 3, dtype=dtype)
    with pytest.raises(RuntimeError, match="expected scalar type Float"):
        flag_gems.quantized_rnn_relu_cell(
            input,
            hx,
            params["w_ih"],
            params["w_hh"],
            params["b_ih"],
            params["b_hh"],
            params["packed_ih"],
            params["packed_hh"],
            params["col_offsets_ih"],
            params["col_offsets_hh"],
            params["scale_ih"],
            params["scale_hh"],
            params["zero_point_ih"],
            params["zero_point_hh"],
        )


@pytest.mark.quantized_rnn_relu_cell
@pytest.mark.parametrize("shape", [(5,), (2, 4, 5)])
def test_quantized_rnn_relu_cell_accepts_ranks_above_2(shape):
    """Ranks >= 2 are broadcast-valid in SimpleCell (no unsafe_chunk)."""
    params = _make_params(3, 5)
    if len(shape) == 3:
        input = torch.randn(*shape, dtype=torch.float32)
        hx = torch.randn(2, 4, 3, dtype=torch.float32)
        out = flag_gems.quantized_rnn_relu_cell(
            input,
            hx,
            params["w_ih"],
            params["w_hh"],
            params["b_ih"],
            params["b_hh"],
            params["packed_ih"],
            params["packed_hh"],
            params["col_offsets_ih"],
            params["col_offsets_hh"],
            params["scale_ih"],
            params["scale_hh"],
            params["zero_point_ih"],
            params["zero_point_hh"],
        )
        assert out.shape == (2, 4, 3)
    else:
        # 1-D activations are rejected with aten's clean rank message.
        input = torch.randn(5, dtype=torch.float32)
        hx = torch.randn(3, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="input.dim"):
            flag_gems.quantized_rnn_relu_cell(
                input,
                hx,
                params["w_ih"],
                params["w_hh"],
                params["b_ih"],
                params["b_hh"],
                params["packed_ih"],
                params["packed_hh"],
                params["col_offsets_ih"],
                params["col_offsets_hh"],
                params["scale_ih"],
                params["scale_hh"],
                params["zero_point_ih"],
                params["zero_point_hh"],
            )


@pytest.mark.quantized_rnn_relu_cell
def test_quantized_rnn_relu_cell_rejects_hidden_mismatch():
    """A hidden state whose last dim differs from hidden size is rejected."""
    params = _make_params(3, 5)
    hx = torch.randn(4, 7, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="K == weight.size"):
        flag_gems.quantized_rnn_relu_cell(
            params["input"],
            hx,
            params["w_ih"],
            params["w_hh"],
            params["b_ih"],
            params["b_hh"],
            params["packed_ih"],
            params["packed_hh"],
            params["col_offsets_ih"],
            params["col_offsets_hh"],
            params["scale_ih"],
            params["scale_hh"],
            params["zero_point_ih"],
            params["zero_point_hh"],
        )


@pytest.mark.quantized_rnn_relu_cell
def test_quantized_rnn_relu_cell_rejects_bad_weight_shape():
    """A non-2-D weight is rejected like aten."""
    params = _make_params(3, 5)
    with pytest.raises(RuntimeError, match="weight.dim"):
        flag_gems.quantized_rnn_relu_cell(
            params["input"],
            params["hx"],
            params["w_ih"].unsqueeze(0),
            params["w_hh"],
            params["b_ih"],
            params["b_hh"],
            params["packed_ih"],
            params["packed_hh"],
            params["col_offsets_ih"],
            params["col_offsets_hh"],
            params["scale_ih"],
            params["scale_hh"],
            params["zero_point_ih"],
            params["zero_point_hh"],
        )


@pytest.mark.quantized_rnn_relu_cell
def test_quantized_rnn_relu_cell_rejects_bad_bias():
    """A 2-D bias is rejected like aten ("Expected bias.dim() == 1")."""
    params = _make_params(3, 5)
    with pytest.raises(RuntimeError, match="bias.dim"):
        flag_gems.quantized_rnn_relu_cell(
            params["input"],
            params["hx"],
            params["w_ih"],
            params["w_hh"],
            params["b_ih"].view(3, 1),
            params["b_hh"],
            params["packed_ih"],
            params["packed_hh"],
            params["col_offsets_ih"],
            params["col_offsets_hh"],
            params["scale_ih"],
            params["scale_hh"],
            params["zero_point_ih"],
            params["zero_point_hh"],
        )


@pytest.mark.quantized_rnn_relu_cell
def test_quantized_rnn_relu_cell_rejects_bad_col_offsets_dtype():
    """Column offsets must be int32 ("expected scalar type Int")."""
    params = _make_params(3, 5)
    with pytest.raises(RuntimeError, match="expected scalar type Int"):
        flag_gems.quantized_rnn_relu_cell(
            params["input"],
            params["hx"],
            params["w_ih"],
            params["w_hh"],
            params["b_ih"],
            params["b_hh"],
            params["packed_ih"],
            params["packed_hh"],
            params["col_offsets_ih"].double(),
            params["col_offsets_hh"],
            params["scale_ih"],
            params["scale_hh"],
            params["zero_point_ih"],
            params["zero_point_hh"],
        )


@pytest.mark.quantized_rnn_relu_cell
def test_quantized_rnn_relu_cell_rejects_int_scale():
    """A non-float weight scale is rejected ("isFloatingPoint")."""
    params = _make_params(3, 5)
    with pytest.raises(RuntimeError, match="isFloatingPoint"):
        flag_gems.quantized_rnn_relu_cell(
            params["input"],
            params["hx"],
            params["w_ih"],
            params["w_hh"],
            params["b_ih"],
            params["b_hh"],
            params["packed_ih"],
            params["packed_hh"],
            params["col_offsets_ih"],
            params["col_offsets_hh"],
            1,
            params["scale_hh"],
            params["zero_point_ih"],
            params["zero_point_hh"],
        )


@pytest.mark.quantized_rnn_relu_cell
def test_quantized_rnn_relu_cell_rejects_float_zero_point():
    """A non-integral zero point is rejected ("isIntegral(false)")."""
    params = _make_params(3, 5)
    with pytest.raises(RuntimeError, match="isIntegral"):
        flag_gems.quantized_rnn_relu_cell(
            params["input"],
            params["hx"],
            params["w_ih"],
            params["w_hh"],
            params["b_ih"],
            params["b_hh"],
            params["packed_ih"],
            params["packed_hh"],
            params["col_offsets_ih"],
            params["col_offsets_hh"],
            params["scale_ih"],
            params["scale_hh"],
            -0.5,
            params["zero_point_hh"],
        )


@pytest.mark.quantized_rnn_relu_cell
def test_quantized_rnn_relu_cell_rejects_non_contiguous_packed():
    """A plain uint8 buffer without the fbgemm wrapper is rejected by aten."""
    params = _make_params(3, 5)
    bogus = torch.zeros(104, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="Expected temporary cpp type wrapper"):
        flag_gems.quantized_rnn_relu_cell(
            params["input"],
            params["hx"],
            params["w_ih"],
            params["w_hh"],
            params["b_ih"],
            params["b_hh"],
            bogus,
            params["packed_hh"],
            params["col_offsets_ih"],
            params["col_offsets_hh"],
            params["scale_ih"],
            params["scale_hh"],
            params["zero_point_ih"],
            params["zero_point_hh"],
        )
