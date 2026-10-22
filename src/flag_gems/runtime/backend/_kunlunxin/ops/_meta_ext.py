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

"""Optional host-only C++ bodies for pure-metadata in-place ops.

Metadata-only operators (``unsqueeze_`` and friends) do no device work; their
entire cost is the O(ndim) Python<->ATen conversion of the size/stride
vectors.  When the benchmark drives ndim into the thousands that conversion is
several times the native kernel (measured: 496 us vs 128 us at ndim=6600),
which no amount of Python tuning can recover -- see ``csrc/meta_ext.cpp``.

Enabled by default; set ``FLAGGEMS_KUNLUNXIN_META_EXT=0`` to force the Python
implementation.  The extension is compiled on first use by torch's extension
builder (cached under ``$TORCH_EXTENSIONS_DIR`` or ``~/.cache/torch_extensions``)
and any failure -- no compiler, no torch headers, wrong toolchain -- silently
falls back to the Python implementation.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_SOURCE = Path(__file__).resolve().parent.parent / "csrc" / "meta_ext.cpp"

_loaded = False
_ext = None


def meta_ext():
    """Return the compiled helper module, or ``None`` if unavailable."""
    global _loaded, _ext
    if _loaded:
        return _ext
    _loaded = True
    if os.environ.get("FLAGGEMS_KUNLUNXIN_META_EXT", "1") != "1":
        return None
    try:
        from torch.utils.cpp_extension import load

        _ext = load(
            name="fg_kunlunxin_meta_ext",
            sources=[str(_SOURCE)],
            extra_cflags=["-O3"],
            with_cuda=False,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - the Python path is always valid
        logger.warning(
            "kunlunxin meta C++ extension unavailable, using Python path: %s", exc
        )
        _ext = None
    return _ext
