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
def _seed_random():
    torch.manual_seed(0)


@pytest.mark.huber_loss_out
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (0, 3)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_huber_loss_out(shape, dtype, reduction, delta):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    reference = torch.ops.aten.huber_loss(
        utils.to_reference(inp, True),
        utils.to_reference(target, True),
        reduction,
        delta,
    )
    expected_shape = shape if reduction == 0 else ()
    out = torch.empty(expected_shape, dtype=dtype, device=flag_gems.device)
    pointer = out.data_ptr()

    with flag_gems.use_gems():
        result = torch.ops.aten.huber_loss.out(
            inp, target, reduction, delta, out=out
        )

    assert result is out
    assert out.data_ptr() == pointer
    utils.gems_assert_close(
        out,
        reference,
        dtype,
        equal_nan=True,
        reduce_dim=max(inp.numel(), 1) if reduction != 0 else 1,
    )


@pytest.mark.huber_loss_out
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_huber_loss_out_resizes_wrong_shape(reduction):
    torch.manual_seed(0)
    inp = torch.randn((4, 7), device=flag_gems.device)
    target = torch.randn_like(inp)
    reference = torch.ops.aten.huber_loss(
        utils.to_reference(inp, True), utils.to_reference(target, True), reduction, 1.0
    )
    out = torch.empty((1,), device=flag_gems.device)
    with flag_gems.use_gems():
        result = torch.ops.aten.huber_loss.out(
            inp, target, reduction, 1.0, out=out
        )
    assert result is out
    assert out.shape == reference.shape
    utils.gems_assert_close(out, reference, torch.float32)


@pytest.mark.huber_loss_out
def test_huber_loss_out_rejects_unsafe_cast_and_invalid_delta():
    inp = torch.randn((2, 3), device=flag_gems.device)
    target = torch.randn_like(inp)
    out = torch.empty((2, 3), dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.ops.aten.huber_loss.out(inp, target, 0, 1.0, out=out)

    float_out = torch.empty_like(inp)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.ops.aten.huber_loss.out(
                inp, target, 0, -1.0, out=float_out
            )
