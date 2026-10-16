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


_SYM_NUMEL_SHAPES = [(), (0,), (2, 3), (10, 20, 30), (100,), (1, 2, 3, 4)]


def _make_input(shape, dtype):
    torch.manual_seed(0)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=flag_gems.device).to(dtype)
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return torch.randint(-10, 10, shape, device=flag_gems.device).to(dtype)


def _assert_numel_result(res_out, ref_out):
    assert isinstance(res_out, int)
    utils.gems_assert_equal(torch.tensor(res_out), torch.tensor(ref_out))


@pytest.mark.sym_numel
@pytest.mark.parametrize("shape", _SYM_NUMEL_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + [torch.bool]
)
def test_sym_numel(shape, dtype, caplog):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.ops.aten.sym_numel(ref_inp)
    with caplog.at_level("DEBUG", logger="flag_gems.ops.sym_numel"):
        res_out = flag_gems.sym_numel(inp)

    assert f"{utils.gems_log_prefix(flag_gems.sym_numel)} SYM_NUMEL" in caplog.text
    _assert_numel_result(res_out, ref_out)


@pytest.mark.sym_numel
def test_sym_numel_noncontiguous():
    torch.manual_seed(0)
    inp = torch.randn((3, 5, 7), device=flag_gems.device).transpose(0, 2)
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.ops.aten.sym_numel(ref_inp)
    res_out = flag_gems.sym_numel(inp)

    _assert_numel_result(res_out, ref_out)
