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


@pytest.mark.special_modified_bessel_k0_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_modified_bessel_k0_out(shape, dtype):
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("enflame doesn't support fp64")
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_x = utils.to_reference(x, True)
    out_ref = torch.empty_like(ref_x)
    ref_out = torch.ops.aten.special_modified_bessel_k0.out(ref_x, out=out_ref)
    out_act = torch.empty_like(x)
    out_ptr = out_act.data_ptr()
    act_out = flag_gems.special_modified_bessel_k0_out(x, out=out_act)
    assert act_out is out_act
    assert out_act.data_ptr() == out_ptr
    utils.gems_assert_close(act_out, ref_out, dtype)
    utils.gems_assert_close(out_act, out_ref, dtype)
