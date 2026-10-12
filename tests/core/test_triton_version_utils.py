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
#
import builtins
from types import SimpleNamespace

import pytest

from flag_gems.utils import triton_version_utils as utils


@pytest.mark.parametrize("probe", ["has_triton_tle", "has_tle_device_mesh"])
@pytest.mark.parametrize("error", [ImportError, OSError])
def test_unavailable_tle_runtime(monkeypatch, probe, error):
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "triton.experimental.tle.language":
            raise error("libcudart.so.12: cannot open shared object file")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(utils, "HAS_TLE", True)
    monkeypatch.setattr(builtins, "__import__", unavailable)
    assert not getattr(utils, probe)()


@pytest.mark.parametrize("probe", ["has_triton_tle", "has_tle_device_mesh"])
def test_tle_probe_preserves_unexpected_errors(monkeypatch, probe):
    original_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "triton.experimental.tle.language":
            raise RuntimeError("unexpected TLE initialization failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(utils, "HAS_TLE", True)
    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(RuntimeError, match="unexpected TLE initialization"):
        getattr(utils, probe)()


@pytest.mark.parametrize("supported_vendor", [False, True])
@pytest.mark.parametrize("has_mesh", [False, True])
def test_loadable_tle_capabilities(monkeypatch, supported_vendor, has_mesh):
    original_import = builtins.__import__
    language = SimpleNamespace(device_mesh=object()) if has_mesh else SimpleNamespace()
    package = SimpleNamespace(
        experimental=SimpleNamespace(tle=SimpleNamespace(language=language))
    )

    def available(name, *args, **kwargs):
        if name == "triton.experimental.tle.language":
            return package
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(utils, "is_support_vendor", lambda: supported_vendor)
    monkeypatch.setattr(builtins, "__import__", available)
    supported = utils.has_triton_tle()
    assert supported is supported_vendor
    monkeypatch.setattr(utils, "HAS_TLE", supported)
    assert utils.has_tle_device_mesh() is (supported_vendor and has_mesh)
