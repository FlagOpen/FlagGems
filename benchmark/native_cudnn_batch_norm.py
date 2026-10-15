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
"""Check native .out lifetime safety without importing FlagGems in the child."""

import json
import logging
import signal
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
_PROBE_REPETITIONS = 32

_PROBE_STARTED = "NATIVE_CUDNN_BATCH_NORM_OUT_STARTED"
_PROBE_COMPLETED = "NATIVE_CUDNN_BATCH_NORM_OUT_COMPLETED"


def probe_native_out(shapes, dtypes, device):
    return subprocess.run(
        [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve())],
        input=json.dumps(
            {
                "shapes": shapes,
                "dtypes": [str(dtype).removeprefix("torch.") for dtype in dtypes],
                "device": str(device),
            }
        ),
        capture_output=True,
        text=True,
        timeout=120,
    )


def native_out_skip_reason(result):
    if result.returncode == 0 and _PROBE_COMPLETED in result.stdout:
        return None
    # Do not classify an import/device setup failure as the known native bug.
    if _PROBE_STARTED in result.stdout:
        if result.returncode in (-signal.SIGSEGV, -signal.SIGABRT):
            return (
                "native cudnn_batch_norm.out lifetime probe terminated with "
                f"{signal.Signals(-result.returncode).name}; no valid native baseline"
            )
        if result.returncode != 0 and "pyobject_preservation.cpp" in result.stderr:
            return (
                "native cudnn_batch_norm.out failed PyTorch object preservation; "
                "no valid native baseline"
            )
    raise RuntimeError(
        f"native cudnn_batch_norm.out probe failed ({result.returncode})\n"
        f"{result.stdout}\n{result.stderr}"
    )


def _run_probe(config):
    import gc

    import torch

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logger.info("torch=%s cuda=%s", torch.__version__, torch.version.cuda)
    device = config["device"]
    for dtype_name in config["dtypes"]:
        dtype = getattr(torch, dtype_name)
        param_dtype = torch.float32 if dtype == torch.float16 else dtype
        for shape in config["shapes"]:
            for training in (True, False):
                x = torch.randn(shape, dtype=dtype, device=device)
                w = torch.ones(shape[1], dtype=param_dtype, device=device)
                b = torch.zeros_like(w)
                rm, rv = b.clone(), w.clone()
                outputs = dict(
                    out0=torch.empty_like(x),
                    out1=torch.empty(
                        shape[1] if training else 0, dtype=param_dtype, device=device
                    ),
                    out2=torch.empty(
                        shape[1] if training else 0, dtype=param_dtype, device=device
                    ),
                    out3=torch.empty(0, dtype=torch.uint8, device=device),
                )
                logger.info("%s %s %s %s", _PROBE_STARTED, dtype_name, shape, training)
                for _ in range(_PROBE_REPETITIONS):
                    # Match benchmark timing: discard each returned tuple.
                    torch.ops.aten.cudnn_batch_norm.out(
                        x, w, b, rm, rv, training, 0.1, 1e-5, **outputs
                    )
                torch.cuda.synchronize(device)
                del outputs, x, w, b, rm, rv
                gc.collect()
    assert "flag_gems" not in sys.modules
    logger.info(_PROBE_COMPLETED)
    # Normal interpreter shutdown is part of the probe; never use os._exit().


if __name__ == "__main__":
    _run_probe(json.load(sys.stdin))
