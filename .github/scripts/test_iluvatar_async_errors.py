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

"""Run real pytest.raises checks; isolate each poisoned device context."""

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


# Old discovery probes are preserved in earlier commits and CI artifacts.
# This suite verifies only the candidate remedy and its generalization.
# Network-related runner retries preserve the same candidate and test matrix.
DTYPES = ["float16", "float32", "bfloat16", "float64"]
ORDERS = ["backward_first", "forward_first", "forward_only_first", "mixed"]
CASES = [
    (dtype, count, order)
    for dtype in DTYPES
    for count in (1, 2, 513)
    for order in ORDERS
]
CASES += [("float32", count, "forward_first") for count in (255, 256, 257)]


def _run_candidate(dtype_name, count, order, stream_mode="default"):
    import contextlib
    import importlib

    import torch

    import flag_gems

    if dtype_name == "float64" and not flag_gems.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    if dtype_name == "bfloat16" and not flag_gems.runtime.device.support_bf16:
        pytest.skip("Device does not support bfloat16")
    forward = importlib.import_module(
        "flag_gems.runtime.backend._iluvatar.ops.embedding_bag"
    )
    backward = importlib.import_module(
        "flag_gems.runtime.backend._iluvatar.ops.embedding_bag_backward"
    )
    dtype = getattr(torch, dtype_name)
    observed = []
    binaries = []
    original_check = backward._check_native_error

    def record_check(error, length):
        compiled = original_check(error, length)
        binaries.append((length, compiled.hash))
        return compiled

    def record_backward_check(error, length):
        observed.append(length)
        return record_check(error, length)

    forward._check_native_error = record_check
    backward._check_native_error = record_backward_check
    # Initialize on the default stream for the cross-stream dependency case.
    if stream_mode == "cross_stream":
        seed_weight = torch.ones((8, 33), device="cuda")
        seed_indices = torch.zeros(2, device="cuda", dtype=torch.int64)
        seed_offsets = torch.zeros(1, device="cuda", dtype=torch.int64)
        forward._embedding_bag(seed_weight, seed_indices, seed_offsets, mode=1)
    context = (
        torch.cuda.stream(torch.cuda.Stream())
        if stream_mode != "default"
        else contextlib.nullcontext()
    )
    with context:
        samples = count * 256
        weight = torch.ones((8, 33), dtype=dtype, device="cuda")
        indices = torch.ones(samples, dtype=torch.int64, device="cuda")
        offsets = torch.tensor([0, samples // 2], dtype=torch.int64, device="cuda")
        mapping = torch.arange(samples, device="cuda") // (samples // 2)
        sizes = torch.full((2,), samples // 2, dtype=torch.int64, device="cuda")
        maximum = torch.empty(0, dtype=torch.int64, device="cuda")
        grad = torch.ones((2, 33), dtype=dtype, device="cuda")
        if order != "backward_first":
            fn = (
                forward._embedding_bag_forward_only
                if order == "forward_only_first"
                else forward._embedding_bag
            )
            if order == "mixed":
                for warm_dtype in (torch.float16, torch.float32, torch.bfloat16):
                    if (
                        warm_dtype == torch.bfloat16
                        and not flag_gems.runtime.device.support_bf16
                    ):
                        continue
                    fn(weight.to(warm_dtype), indices, offsets, mode=1)
            _, mapping, sizes, maximum = fn(weight, indices, offsets, mode=1)
        args = [grad, indices, offsets, mapping, sizes, maximum, 8, False, 1, False]
        result = backward._embedding_bag_backward(*args)
        assert result.shape == (8, 33)
        assert observed == [count], f"Unexpected validation flag lengths: {observed}"
        assert len(forward._NATIVE_ASSERT_STATES) == 1
        print(f"ASSERT_BINARIES {binaries}", flush=True)
        assert binaries[0][0] == 1, "The sole seed must have one flag"
        assert len({binary for _, binary in binaries}) == 1, binaries
        # Put the bad index in the last validation block, not always block zero.
        indices[-1] = 8
        print(
            f"CANDIDATE dtype={dtype_name} flags={count} order={order} stream={stream_mode}",
            flush=True,
        )
        with pytest.raises(RuntimeError, match="(?i)assert") as caught:
            result = backward._embedding_bag_backward(*args)
            torch.cuda.synchronize()
        print(
            f"CAUGHT {type(caught.value).__module__}.{type(caught.value).__name__}",
            flush=True,
        )


@pytest.mark.parametrize("dtype,count,order", CASES)
def test_candidate_backward(request, dtype, count, order):
    if _isolated(request):
        return
    _run_candidate(dtype, count, order)


@pytest.mark.parametrize("stream_mode", ["new_stream", "cross_stream"])
def test_candidate_stream_order(request, stream_mode):
    if _isolated(request):
        return
    _run_candidate("float32", 257, "forward_first", stream_mode)
