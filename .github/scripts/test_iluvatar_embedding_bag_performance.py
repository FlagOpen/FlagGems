# Copyright 2026 FlagOS Contributors
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

"""Isolated valid-input performance diagnostics; never a production bypass."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

CHILD = "IX_PYTEST_ERROR_CHILD"


def _isolated(request):
    if os.environ.get(CHILD) == "1":
        return False
    # A fresh process keeps one failed device assertion from poisoning later probes.
    # Keep each parametrized case isolated, including the successful controls.
    node = str(Path(__file__).resolve()) + "::" + request.node.name
    env = dict(os.environ, **{CHILD: "1", "TRITON_DEBUG": "0"})
    command = [sys.executable, "-m", "pytest", "-c", "/dev/null", "-s", "-vv", node]
    with subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            output, _ = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            pytest.fail("Child pytest timed out:\n" + output)
    print(output, flush=True)
    assert process.returncode == 0, f"Child pytest exit={process.returncode}\n{output}"
    if "1 skipped" in output:
        pytest.skip("Isolated device capability skip; see child stdout")
    assert "1 passed" in output, output
    return True


SHAPES = [(8, 16, 128, 4), (1024, 64, 16384, 64)]


@pytest.mark.parametrize("variant", ["_embedding_bag", "_embedding_bag_forward_only"])
@pytest.mark.parametrize("dtype_name", ["float16", "float32", "bfloat16"])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("shape", SHAPES, ids=["small", "large"])
def test_forward_performance(request, variant, dtype_name, mode, shape):
    if _isolated(request):
        return
    import cProfile
    import importlib
    import inspect
    import io
    import json
    import pstats
    import statistics
    import time
    import warnings

    import torch
    import triton
    import triton.language as tl

    import flag_gems

    if dtype_name == "bfloat16" and not flag_gems.runtime.device.support_bf16:
        pytest.skip("Device does not support bfloat16")
    backend = importlib.import_module(
        "flag_gems.runtime.backend._iluvatar.ops.embedding_bag"
    )
    path = Path("../ix-results/performance-probes") / request.node.name
    path.mkdir(parents=True, exist_ok=True)
    bags, dim, rows, length = shape
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(1234)
    weight = torch.randn((rows, dim), dtype=dtype, device="cuda")
    indices = torch.randint(1, rows, (bags * length,), device="cuda")
    offsets = torch.arange(0, bags * length, length, device="cuda")
    args = (weight, indices, offsets, False, mode, False, None, False, 0)
    native = getattr(torch.ops.aten, variant).default
    gems = getattr(backend, variant)
    report = {
        "variant": variant,
        "dtype": dtype_name,
        "mode": mode,
        "shape": shape,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "measurements": {},
    }

    def save():
        (path / "results.json").write_text(json.dumps(report, indent=2))

    (path / "triton-timers.py.txt").write_text(
        inspect.getsource(triton.testing.do_bench)
        + "\n"
        + inspect.getsource(triton.testing.do_bench_cudagraph)
    )
    # Measure first call separately; it includes lazy setup/compilation and is
    # never mixed into steady-state comparisons.
    for name, function in (("native", native), ("candidate", gems)):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            function(*args)
            torch.cuda.synchronize()
        report[name + "_first_call_ms"] = (time.perf_counter() - begin) * 1000
        report[name + "_warnings"] = [str(w.message) for w in caught]
        if any(
            "fallback" in str(w.message).lower().replace("fall back", "fallback")
            and "cpu" in str(w.message).lower()
            for w in caught
        ):
            report["status"] = "CPU_FALLBACK_UNMEASURED"
            save()
            pytest.skip("Native/device CPU fallback warning; saved warning evidence")
    report["runtime_flags"] = {
        key: os.environ.get(key)
        for key in (
            "CUDA_LAUNCH_BLOCKING",
            "TRITON_DEBUG",
            "TRITON_INTERPRET",
            "TORCH_USE_CUDA_DSA",
        )
    }
    report["device"] = torch.cuda.get_device_name()
    for name, obj in (
        ("jit-run", triton.runtime.JITFunction.run),
        ("forward-impl", backend._generic_impl),
    ):
        (path / f"{name}-source.py.txt").write_text(inspect.getsource(obj))
    save()

    @triton.jit
    def empty_launch(buffer):
        value = tl.load(buffer)
        tl.store(buffer, value)

    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    flags = torch.zeros(bags, dtype=torch.int32, device="cuda")
    functions = {
        "native": lambda: native(*args),
        "candidate": lambda: gems(*args),
        "checker_only": lambda: backend._check_native_error(flags, bags),
        "empty_debug_false": lambda: empty_launch[(1,)](zero, debug=False),
        "empty_debug_true": lambda: empty_launch[(1,)](zero, debug=True),
    }

    def measure(function):
        for _ in range(5):
            function()
        torch.cuda.synchronize()
        values = {"triton_ms": [], "batch_event_ms": [], "wall_ms": [], "submit_ms": []}
        for _ in range(3):
            values["triton_ms"].append(
                triton.testing.do_bench(
                    function, warmup=20, rep=60, return_mode="median"
                )
            )
            count = 50
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            torch.cuda.synchronize()
            begin = time.perf_counter()
            start.record()
            enqueue = time.perf_counter()
            for _ in range(count):
                function()
            submitted = time.perf_counter()
            end.record()
            end.synchronize()
            finish = time.perf_counter()
            values["batch_event_ms"].append(start.elapsed_time(end) / count)
            values["wall_ms"].append((finish - begin) * 1000 / count)
            values["submit_ms"].append((submitted - enqueue) * 1000 / count)
        return {
            "raw": values,
            "median": {key: statistics.median(v) for key, v in values.items()},
        }

    for name, function in functions.items():
        report["measurements"][name] = measure(function)
        save()
    # Diagnostic ablations use only in-range, prevalidated inputs. Neither is
    # installed into operator-source or used as an accepted implementation.
    for name, attribute in (
        ("without_check", "_check_native_error"),
        ("without_prepare", "_prepare_native_assert"),
    ):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(backend, attribute, lambda *args: None)
            report["measurements"][name] = measure(functions["candidate"])
        save()

    # Audit Python-visible synchronizations/readbacks only during the op call;
    # timer synchronization is outside this region. Profiler traces below cover
    # lower-level runtime activity that these wrappers cannot observe.
    calls = []

    def audit(name, original):
        def wrapped(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        return wrapped

    with pytest.MonkeyPatch.context() as patch:
        for name in ("item", "tolist", "cpu", "numpy"):
            patch.setattr(
                torch.Tensor, name, audit("Tensor." + name, getattr(torch.Tensor, name))
            )
        patch.setattr(
            torch.cuda, "synchronize", audit("cuda.synchronize", torch.cuda.synchronize)
        )
        patch.setattr(
            torch.cuda.Event,
            "synchronize",
            audit("Event.synchronize", torch.cuda.Event.synchronize),
        )
        patch.setattr(
            torch.cuda.Stream,
            "synchronize",
            audit("Stream.synchronize", torch.cuda.Stream.synchronize),
        )
        functions["candidate"]()
    report["candidate_python_sync_readback_calls"] = calls
    torch.cuda.synchronize()
    save()

    for name in ("native", "candidate", "checker_only"):
        function = functions[name]
        profile = cProfile.Profile()
        profile.enable()
        for _ in range(30):
            function()
        profile.disable()
        torch.cuda.synchronize()
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(60)
        (path / f"{name}-python-profile.txt").write_text(stream.getvalue())
        # Optional profiler capability is reported explicitly, never confused
        # with a successful device trace when CoreX lacks Kineto support.
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for _ in range(5):
                    function()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(path / f"{name}-trace.json"))
            (path / f"{name}-profiler.txt").write_text(
                prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=60)
            )
        except Exception as error:
            report[name + "_profiler_error"] = repr(error)
            save()
            raise

    # Graph replay separates device work from repeated Python dispatch. Run it
    # last: a failed capture can invalidate the device context on some runtimes.
    for name in (
        "native",
        "candidate",
        "checker_only",
        "empty_debug_false",
        "empty_debug_true",
    ):
        try:
            report["measurements"][name]["graph_ms"] = (
                triton.testing.do_bench_cudagraph(
                    functions[name], rep=60, return_mode="median"
                )
            )
        except Exception as error:
            report["graph_error"] = {"stage": name, "error": repr(error)}
            save()
            break
        save()
    print(json.dumps(report, indent=2), flush=True)
