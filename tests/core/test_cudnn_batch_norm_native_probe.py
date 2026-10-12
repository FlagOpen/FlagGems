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
import signal
from subprocess import CompletedProcess

import pytest

from benchmark.native_cudnn_batch_norm import (
    _PROBE_COMPLETED,
    _PROBE_STARTED,
    native_out_skip_reason,
)


def test_native_out_probe_requires_normal_shutdown():
    assert native_out_skip_reason(CompletedProcess([], 0, _PROBE_COMPLETED, "")) is None
    # Even completed calls are not a success if interpreter cleanup crashes.
    result = CompletedProcess(
        [], -signal.SIGSEGV, _PROBE_STARTED + _PROBE_COMPLETED, ""
    )
    assert "SIGSEGV" in native_out_skip_reason(result)


@pytest.mark.parametrize("signal_code", [signal.SIGSEGV, signal.SIGABRT])
def test_native_out_probe_recognizes_native_crash(signal_code):
    result = CompletedProcess([], -signal_code, _PROBE_STARTED, "native crash")
    assert "no valid native baseline" in native_out_skip_reason(result)


def test_native_out_probe_recognizes_object_preservation_failure():
    result = CompletedProcess([], 1, _PROBE_STARTED, "pyobject_preservation.cpp:45")
    assert "object preservation" in native_out_skip_reason(result)


@pytest.mark.parametrize(
    "code,stdout,stderr",
    [
        (-signal.SIGSEGV, "", "import crash"),
        (-signal.SIGKILL, _PROBE_STARTED, ""),
        (1, _PROBE_STARTED, "CUDA out of memory"),
        (1, _PROBE_STARTED, "unexpected runtime error"),
        (0, _PROBE_STARTED, ""),
    ],
)
def test_native_out_probe_does_not_hide_unrelated_errors(code, stdout, stderr):
    with pytest.raises(RuntimeError, match="probe failed"):
        native_out_skip_reason(CompletedProcess([], code, stdout, stderr))
