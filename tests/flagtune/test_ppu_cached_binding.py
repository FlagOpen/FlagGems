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

import gc
import importlib
import inspect
import weakref
from types import SimpleNamespace

import pytest
import torch

import flag_gems

lib = importlib.import_module("flag_gems.utils.libentry")

_mm_backend = importlib.import_module(flag_gems.mm.__module__)
pytestmark = pytest.mark.skipif(
    lib.device.vendor_name != "thead"
    or not (lib.major_version == 3 and 3 <= lib.minor_version <= 6)
    or not getattr(_mm_backend, "HAS_PPU_TLE", False),
    reason="requires the tuned PPU cached launch path",
)


class CaptureKey(Exception):
    pass


class Kernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args):
            self.calls.append((grid, args))

        return launch


@pytest.fixture
def entry(monkeypatch):
    monkeypatch.setattr(lib.torch_device_fn, "current_device", lambda: 0)
    e = object.__new__(lib.PPULibEntry)
    e.arg_names = ["A", "N", "alpha", "FLAG", "BLOCK", "HEUR"]
    kinds = (1, 1, 2, 0, 0, 0)
    defaults = (
        inspect._empty,
        inspect._empty,
        1.0,
        False,
        inspect._empty,
        inspect._empty,
    )
    e.signature = inspect.Signature(
        [
            inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for n in e.arg_names
        ]
    )
    e.jit_function = SimpleNamespace(
        params=[
            SimpleNamespace(
                name=n, is_constexpr=k == 0, do_not_specialize=k == 2, default=d
            )
            for n, k, d in zip(e.arg_names, kinds, defaults)
        ]
    )
    e.specialize_indices = [0, 1]
    e.do_not_specialize_indices = [2]
    e.divisibility = 16
    e._has_flagtune_tuner = True
    e._apply_flagtune = lambda device: None
    e.kernel_cache = ({},)
    e._cpu_cache = {}
    e._ppu_launch_plans = {}
    e._ppu_tensor_type = torch.Tensor
    e._ppu_scalar_types = (int, bool, float, str, type(None))
    e._ppu_param_names = tuple(e.arg_names)
    e._ppu_param_kinds = kinds
    e._ppu_param_defaults = defaults
    e._ppu_dtype_names = {}
    return e


def seed_from_generic(e, args, kwargs, hooks=(), block=8):
    def capture(spec, dns, const):
        raise CaptureKey(lib.LibEntry.key(e, spec, dns, const))

    e.key = capture
    with pytest.raises(CaptureKey) as caught:
        lib.LibEntry.run(e, *args, **kwargs)
    del e.key
    key = caught.value.args[0]
    kernel = Kernel()
    e.kernel_cache[0][key] = (
        kernel,
        {"BLOCK": block, "HEUR": 3},
        {"BLOCK": block},
        {"HEUR": 3},
        hooks,
    )
    return key, kernel


def assert_same_call(left, right):
    assert left[0] == right[0]
    assert len(left[1]) == len(right[1])
    for a, b in zip(left[1], right[1]):
        if isinstance(a, torch.Tensor):
            assert a is b
        else:
            assert type(a) is type(b) and a == b


@pytest.mark.parametrize("positional", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_generic_and_cached_binding_agree(entry, positional, dtype):
    values = (torch.arange(8, dtype=dtype), 8, 0.5, True)
    args = values[:positional]
    kwargs = dict(zip(entry.arg_names[positional:4], values[positional:]))
    kwargs["grid"] = lambda m: (m["N"] // m["BLOCK"] + 1,)
    _, kernel = seed_from_generic(entry, args, kwargs)
    lib.LibEntry.run(entry, *args, **kwargs)
    assert entry._run_ppu_cached(args, kwargs) is not None
    assert_same_call(kernel.calls[0], kernel.calls[1])


def test_current_tensors_and_unspecialized_values_are_not_frozen(entry):
    a = torch.ones(8)
    b = torch.zeros(8)
    kwargs = {"grid": (1,)}
    _, kernel = seed_from_generic(entry, (a, 8, 0.25), kwargs)
    entry._run_ppu_cached((a, 8, 0.25), kwargs)
    entry._run_ppu_cached((b, 8, 0.75), kwargs)
    assert kernel.calls[-1][1][0] is b
    assert kernel.calls[-1][1][2] == 0.75


def test_defaults_and_explicit_keyword_values_share_correct_plan(entry):
    a = torch.ones(8)
    kwargs = {"grid": (1,)}
    _, kernel = seed_from_generic(entry, (a, 8), kwargs)
    entry._run_ppu_cached((a, 8), kwargs)
    entry._run_ppu_cached((a, 8), dict(kwargs, alpha=0.3, FLAG=False))
    assert kernel.calls[-1][1][2:] == (0.3, False, 8, 3)


@pytest.mark.parametrize("alpha", [-(2**31) - 1, 2**31, 2**63, 2**64 - 1, 2**64])
def test_unspecialized_integer_width_matches_generic(entry, alpha):
    args = (torch.ones(8), 8, alpha)
    kwargs = {"grid": (1,)}
    _, kernel = seed_from_generic(entry, args, kwargs)
    assert entry._run_ppu_cached(args, kwargs) is not None
    assert kernel.calls[-1][1][2] == alpha


def test_alignment_and_dtype_changes_do_not_reuse_wrong_entry(entry):
    a = torch.ones(8)
    kwargs = {"grid": (1,)}
    seed_from_generic(entry, (a, 8), kwargs)
    assert entry._run_ppu_cached((torch.ones(9)[1:], 8), kwargs) is None
    assert entry._run_ppu_cached((a.to(torch.bfloat16), 8), kwargs) is None


def test_custom_constexpr_uses_generic_fallback(entry):
    marker = object()
    args = (torch.ones(8), 8, 1.0, marker)
    kwargs = {"grid": (1,)}
    _, kernel = seed_from_generic(entry, args, kwargs)
    assert entry._run_ppu_cached(args, kwargs) is None
    entry.run(*args, **kwargs)
    assert kernel.calls[-1][1][3] is marker


def test_hooks_receive_current_arguments(entry):
    seen = []
    a = torch.ones(8)
    b = torch.zeros(8)
    kwargs = {"grid": (1,)}
    seed_from_generic(entry, (a, 8), kwargs, hooks=((seen.append, {"hook_value": 4}),))
    entry._run_ppu_cached((b, 8), kwargs)
    assert seen[-1]["A"] is b and seen[-1]["hook_value"] == 4


def test_replaced_kernel_cache_entry_refreshes_launch_constants(entry):
    args = (torch.ones(8), 8)
    kwargs = {"grid": (1,)}
    seed_from_generic(entry, args, kwargs, block=8)
    entry._run_ppu_cached(args, kwargs)
    _, kernel = seed_from_generic(entry, args, kwargs, block=16)
    entry._run_ppu_cached(args, kwargs)
    assert kernel.calls[-1][1][-2] == 16


def test_launch_plan_does_not_retain_input_tensors(entry):
    def launch():
        a = torch.ones(8)
        ref = weakref.ref(a)
        args = (a, 8)
        kwargs = {"grid": (1,)}
        _, kernel = seed_from_generic(entry, args, kwargs)
        entry._run_ppu_cached(args, kwargs)
        kernel.calls.clear()
        return ref

    ref = launch()
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "sliced", "singleton", "empty", "offset"]
)
def test_dense_transpose_detection_matches_torch(kind):
    import flag_gems

    mm = importlib.import_module(flag_gems.mm.__module__)
    candidates = {
        "dense": torch.empty(3, 4),
        "transpose": torch.empty(4, 3).t(),
        "sliced": torch.empty(8, 6)[::2, ::2],
        "singleton": torch.empty(1, 4).t(),
        "empty": torch.empty(0, 4).t(),
        "offset": torch.empty(20)[2:14].view(4, 3).t(),
    }
    value = candidates[kind]
    expected = (
        value.ndim == 2 and not value.is_contiguous() and value.t().is_contiguous()
    )
    assert mm._is_transposed_contiguous_2d(value) == expected


def test_split_reducer_uses_this_launch_metadata_without_rebuilding_configs(
    monkeypatch,
):
    from contextlib import nullcontext

    import flag_gems

    mm = importlib.import_module(flag_gems.mm.__module__)
    selected = [2, 4, 2]
    seen = []

    class Main:
        best_config = SimpleNamespace(kwargs={"SPLIT_K": 8})

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                return self, {"SPLIT_K": selected.pop(0)}

            return launch

    class Reduce:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                seen.append(kwargs["SPLIT_K"])

            return launch

    def rebuild():
        raise AssertionError("candidate catalog rebuilt on hot path")

    monkeypatch.setattr(mm, "mm_split_k_kernel_ppu", Main())
    monkeypatch.setattr(mm, "mm_split_k_reduce_kernel_ppu", Reduce())
    monkeypatch.setattr(mm, "_ppu_split_k_configs", rebuild)
    monkeypatch.setattr(mm, "_ppu_device_context", lambda tensor: nullcontext())
    a = torch.empty(4, 2048, dtype=torch.bfloat16)
    b = torch.empty(2048, 64, dtype=torch.bfloat16)
    out = torch.empty(4, 64, dtype=torch.bfloat16)
    for _ in range(3):
        assert mm._run_split_k_mm(a, b, out) is out
    assert seen == [2, 4, 2]


def test_offline_benchmark_clear_also_releases_ppu_launch_plans(entry):
    args = (torch.ones(8), 8)
    kwargs = {"grid": (1,)}
    seed_from_generic(entry, args, kwargs)
    entry._run_ppu_cached(args, kwargs)
    assert entry._ppu_launch_plans
    lib.clear_libentry_dispatch_cache(entry)
    assert not any(entry.kernel_cache)
    assert not entry._ppu_launch_plans
    assert entry._run_ppu_cached(args, kwargs) is None


@pytest.mark.parametrize("vendor", ["thead", "nvidia", "cpu", "hygon", "mthreads"])
@pytest.mark.parametrize("opt_in", [False, True])
def test_cached_binding_requires_explicit_ppu_opt_in(monkeypatch, vendor, opt_in):
    normal = object()
    cached = object()
    monkeypatch.setattr(lib, "device", SimpleNamespace(vendor_name=vendor))
    monkeypatch.setattr(lib, "major_version", 3)
    monkeypatch.setattr(lib, "minor_version", 6)
    monkeypatch.setattr(lib, "LibEntry", lambda fn: normal)
    monkeypatch.setattr(lib, "PPULibEntry", lambda fn: cached)
    result = lib.libentry(ppu_cached=opt_in)(object())
    assert result is (cached if vendor == "thead" and opt_in else normal)


@pytest.mark.parametrize("major,minor", [(2, 3), (3, 2), (3, 7), (4, 0)])
def test_unsupported_triton_uses_generic_binding(monkeypatch, major, minor):
    normal = object()
    monkeypatch.setattr(lib, "device", SimpleNamespace(vendor_name="thead"))
    monkeypatch.setattr(lib, "major_version", major)
    monkeypatch.setattr(lib, "minor_version", minor)
    monkeypatch.setattr(lib, "LibEntry", lambda fn: normal)
    assert lib.libentry(ppu_cached=True)(object()) is normal
