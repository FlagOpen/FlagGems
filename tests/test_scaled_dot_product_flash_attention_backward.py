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

import math

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import set_philox_state

from . import accuracy_utils as utils
from . import conftest as cfg

pytestmark = pytest.mark.skipif(
    cfg.TO_CPU, reason="ATen FlashAttention is unavailable in CPU reference mode"
)

DTYPES = [torch.float16, torch.bfloat16]
CASES = [
    (1, 2, 16, 16, 32),
    (2, 4, 17, 31, 64),
    (1, 1, 65, 33, 128),
]


def _set_seed():
    torch.manual_seed(0)
    set_philox_state(1234567890, 0, torch_device_fn.current_device())


def _make_dense_inputs(batch, heads, q_len, kv_len, head_dim, dtype, is_causal, scale):
    _set_seed()
    query = torch.empty(
        (batch, q_len, heads, head_dim), dtype=dtype, device=flag_gems.device
    ).uniform_(-0.05, 0.05)
    key = torch.empty(
        (batch, kv_len, heads, head_dim), dtype=dtype, device=flag_gems.device
    ).uniform_(-0.05, 0.05)
    value = torch.empty_like(key).uniform_(-0.05, 0.05)
    grad_out = torch.randn_like(query)
    out, logsumexp, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
        query,
        key,
        value,
        None,
        None,
        q_len,
        kv_len,
        0.0,
        is_causal,
        False,
        scale=scale,
    )
    return (
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp.float(),
        None,
        None,
        q_len,
        kv_len,
        0.0,
        is_causal,
        rng_state[0],
        rng_state[1],
    )


def _call(args, scale):
    return torch.ops.aten._scaled_dot_product_flash_attention_backward(
        *args, scale=scale
    )


def _to_reference_args(args):
    result = []
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_floating_point():
            result.append(utils.to_reference(arg, True))
        elif isinstance(arg, torch.Tensor):
            result.append(utils.to_reference(arg, False))
        else:
            result.append(arg)
    return tuple(result)


@pytest.mark.scaled_dot_product_flash_attention_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", CASES)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("scale_mode", ["default", "explicit"])
def test_scaled_dot_product_flash_attention_backward(
    dtype, shape, is_causal, scale_mode
):
    batch, heads, q_len, kv_len, head_dim = shape
    scale = None if scale_mode == "default" else 0.5 / math.sqrt(head_dim)
    args = _make_dense_inputs(
        batch, heads, q_len, kv_len, head_dim, dtype, is_causal, scale
    )
    ref_args = _to_reference_args(args)

    expected = _call(ref_args, scale)
    with flag_gems.use_gems():
        actual = _call(args, scale)

    assert len(actual) == len(expected) == 3
    for result, reference, source in zip(actual, expected, args[1:4]):
        utils.gems_assert_close(result, reference, dtype, equal_nan=True)
        assert result.shape == source.shape
        assert result.dtype == source.dtype
        assert result.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


@pytest.mark.scaled_dot_product_flash_attention_backward
@pytest.mark.parametrize("dtype", DTYPES)
def test_scaled_dot_product_flash_attention_backward_varlen_metadata(dtype):
    _set_seed()
    heads, head_dim = 2, 32
    cum_seq_q = torch.tensor([0, 3, 8], dtype=torch.int32, device=flag_gems.device)
    cum_seq_k = torch.tensor([0, 5, 9], dtype=torch.int32, device=flag_gems.device)
    query = torch.empty((8, heads, head_dim), dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    key = torch.empty((9, heads, head_dim), dtype=dtype, device=flag_gems.device).uniform_(
        -0.05, 0.05
    )
    value = torch.empty_like(key).uniform_(-0.05, 0.05)
    grad_out = torch.randn_like(query)
    out, logsumexp, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
        query,
        key,
        value,
        cum_seq_q,
        cum_seq_k,
        5,
        5,
        0.0,
        False,
        False,
        scale=None,
    )
    args = (
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp.float(),
        cum_seq_q,
        cum_seq_k,
        5,
        5,
        0.0,
        False,
        rng_state[0],
        rng_state[1],
    )
    expected = _call(_to_reference_args(args), None)
    with flag_gems.use_gems():
        actual = _call(args, None)

    for result, reference in zip(actual, expected):
        utils.gems_assert_close(result, reference, dtype, equal_nan=True)


@pytest.mark.scaled_dot_product_flash_attention_backward
@pytest.mark.parametrize("dropout_p", [0.1, 0.25])
def test_scaled_dot_product_flash_attention_backward_dropout_philox(dropout_p):
    _set_seed()
    dtype = torch.float16
    query = torch.randn((1, 16, 2, 32), dtype=dtype, device=flag_gems.device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    grad_out = torch.randn_like(query)
    out, logsumexp, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
        query,
        key,
        value,
        None,
        None,
        16,
        16,
        dropout_p,
        False,
        False,
        scale=None,
    )
    args = (
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp.float(),
        None,
        None,
        16,
        16,
        dropout_p,
        False,
        rng_state[0],
        rng_state[1],
    )
    expected = _call(_to_reference_args(args), None)
    with flag_gems.use_gems():
        actual = _call(args, None)

    for result, reference in zip(actual, expected):
        utils.gems_assert_close(result, reference, dtype, equal_nan=True)


@pytest.mark.scaled_dot_product_flash_attention_backward
def test_scaled_dot_product_flash_attention_backward_rejects_bad_grad_shape():
    args = list(_make_dense_inputs(1, 2, 16, 16, 32, torch.float16, False, None))
    args[0] = torch.randn((1, 15, 2, 32), dtype=torch.float16, device=flag_gems.device)
    with pytest.raises((RuntimeError, ValueError)):
        with flag_gems.use_gems():
            _call(tuple(args), None)


@pytest.mark.scaled_dot_product_flash_attention_backward
def test_scaled_dot_product_flash_attention_backward_rejects_partial_varlen_metadata():
    args = list(_make_dense_inputs(1, 2, 16, 16, 32, torch.float16, False, None))
    args[6] = torch.tensor([0, 16], dtype=torch.int32, device=flag_gems.device)
    args[7] = None
    with pytest.raises((RuntimeError, ValueError)):
        with flag_gems.use_gems():
            _call(tuple(args), None)
