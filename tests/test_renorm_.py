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


@pytest.mark.renorm_
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: renorm_ is not supported"
)
@pytest.mark.parametrize("shape", [(1, 2), (4, 128), (16, 256), (2, 0)])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("p", [1.0, 2.0, 3.5])
@pytest.mark.parametrize("maxnorm", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_renorm_(shape, dim, p, maxnorm, dtype):
    if not shape or dim >= len(shape) or dim < -len(shape):
        pytest.skip("dimension is outside this parametrized rank")
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    original_pointer = inp.data_ptr()
    ref_inp = utils.to_reference(inp.clone(), True)
    reference = torch.ops.aten.renorm_(ref_inp, p, dim, maxnorm)

    with flag_gems.use_gems():
        result = torch.ops.aten.renorm_(inp, p, dim, maxnorm)

    assert result is inp
    assert result.data_ptr() == original_pointer
    utils.gems_assert_close(result, reference, dtype)


@pytest.mark.renorm_
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: renorm_ is not supported"
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_renorm_noncontiguous_(dtype):
    torch.manual_seed(0)
    inp = torch.randn((4, 7, 9), dtype=dtype, device=flag_gems.device).transpose(0, 2)
    original_pointer = inp.data_ptr()
    original_stride = inp.stride()
    reference = torch.ops.aten.renorm_(utils.to_reference(inp.clone(), True), 2.0, 1, 1.0)

    with flag_gems.use_gems():
        result = torch.ops.aten.renorm_(inp, 2.0, 1, 1.0)

    assert result is inp
    assert result.data_ptr() == original_pointer
    assert result.stride() == original_stride
    utils.gems_assert_close(result, reference, dtype)


@pytest.mark.renorm_
@pytest.mark.parametrize("p", [0.0, -1.0])
def test_renorm_rejects_nonpositive_p_(p):
    inp = torch.randn((4, 7), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.ops.aten.renorm_(inp, p, 1, 1.0)


@pytest.mark.renorm_
def test_renorm_rejects_invalid_dim_():
    inp = torch.randn((4, 7), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises((IndexError, RuntimeError)):
        torch.ops.aten.renorm_(inp, 2.0, 2, 1.0)
