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

"""Resolve operator IDs to pytest markers without importing the GPU runtime.

The explicit overrides cover the original 297-operator generated suite's
underscore IDs. Other suites retain their legacy collision-based convention.
"""

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def marker_overrides() -> dict[str, str]:
    path = Path(__file__).resolve().parents[1] / "conf/operator_marker_overrides.json"
    return json.loads(path.read_text())


def operator_marker(op_id: str, all_op_ids: set) -> str:
    override = marker_overrides().get(op_id)
    if override is not None:
        return override
    if not op_id.startswith("_"):
        return op_id
    stripped = op_id.lstrip("_")
    return f"underscore_{stripped}" if stripped in all_op_ids else stripped
