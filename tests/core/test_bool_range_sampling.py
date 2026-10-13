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

"""Check bool range selection using real parametrized pytest items."""

import json

pytest_plugins = ["pytester"]


def _collect(pytester, source):
    pytester.makepyfile(source)
    pytester.makeconftest("""
import json
from pathlib import Path

from tests.bool_range_utils import deduplicate_bool_range_items


def pytest_collection_modifyitems(config, items):
    before = list(items)
    kept, dropped = deduplicate_bool_range_items(items)
    items[:] = kept
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    Path("selection.json").write_text(json.dumps({
        "before": [item.nodeid for item in before],
        "kept": [item.nodeid for item in kept],
        "dropped": [item.nodeid for item in dropped],
        "params": [repr(item.callspec.params) for item in kept],
    }))
""")
    result = pytester.runpytest("--collect-only", "-q", "-p", "no:cacheprovider")
    assert result.ret == 0
    selection = json.loads((pytester.path / "selection.json").read_text())
    kept = set(selection["kept"])
    assert selection["kept"] == [
        nodeid for nodeid in selection["before"] if nodeid in kept
    ]
    assert set(selection["dropped"]) == set(selection["before"]) - kept
    return selection


def test_only_bool_range_cross_product_is_reduced(pytester):
    selection = _collect(
        pytester,
        """
import pytest
import torch
from tests.bool_range_utils import deduplicate_bool_ranges

@deduplicate_bool_ranges
@pytest.mark.parametrize("shape", [(), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.float32])
@pytest.mark.parametrize("value_range", [["-1", "1"], ["0", "1"], ["-1", "0"], ["0", "max"], ["min", "0"]])
def test_grid(shape, dtype, value_range):
    raise AssertionError("collection must not run the test body")
""",
    )
    assert len(selection["before"]) == 30
    assert len(selection["kept"]) == 22
    assert sum("torch.bool" in row for row in selection["params"]) == 2
    assert sum("torch.uint8" in row for row in selection["params"]) == 10
    assert sum("torch.float32" in row for row in selection["params"]) == 10


def test_joint_rows_keep_structures_and_index_widths(pytester):
    selection = _collect(
        pytester,
        """
import pytest
import torch
from tests.bool_range_utils import deduplicate_bool_ranges

ROWS = [(dtype, {"shape": [2, 3], "layout": layout}, interval)
        for dtype in [torch.bool, torch.float32]
        for layout in ["csr", "bsr"]
        for interval in [["-1", "1"], ["0", "max"]]]

@deduplicate_bool_ranges(dtype_arg="values_dtype")
@pytest.mark.parametrize("values_dtype,description,value_range", ROWS)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_sparse(values_dtype, description, value_range, index_dtype):
    pass
""",
    )
    assert len(selection["before"]) == 16
    assert len(selection["kept"]) == 12
    bool_rows = [row for row in selection["params"] if "torch.bool" in row]
    assert len(bool_rows) == 4
    assert sum("csr" in row for row in bool_rows) == 2
    assert sum("bsr" in row for row in bool_rows) == 2
    assert sum("torch.int32" in row for row in bool_rows) == 2
    assert sum("torch.int64" in row for row in bool_rows) == 2


def test_unannotated_and_non_range_tests_are_unchanged(pytester):
    selection = _collect(
        pytester,
        """
import pytest
import torch
from tests.bool_range_utils import deduplicate_bool_ranges

@pytest.mark.parametrize("dtype", [torch.bool])
@pytest.mark.parametrize("value_range", [["-1", "1"], ["0", "max"]])
def test_range_controls_another_operand(dtype, value_range):
    pass

@deduplicate_bool_ranges
@pytest.mark.parametrize("dtype", [torch.bool])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_non_range(dtype, shape):
    pass
""",
    )
    assert selection["kept"] == selection["before"]


def test_parameter_specific_marks_remain_distinct(pytester):
    selection = _collect(
        pytester,
        """
import pytest
import torch
from tests.bool_range_utils import deduplicate_bool_ranges

@deduplicate_bool_ranges
@pytest.mark.parametrize("dtype", [torch.bool])
@pytest.mark.parametrize("value_range", [
    ["-1", "1"],
    pytest.param(["0", "1"], marks=pytest.mark.xfail(reason="distinct contract")),
    pytest.param(["0", "max"], marks=pytest.mark.skip(reason="distinct availability")),
])
def test_marked(dtype, value_range):
    pass
""",
    )
    assert selection["kept"] == selection["before"]


def test_single_range_and_unknown_parameters_are_unchanged(pytester):
    selection = _collect(
        pytester,
        """
import pytest
import torch
from tests.bool_range_utils import deduplicate_bool_ranges

class Unknown:
    __hash__ = None

@deduplicate_bool_ranges
@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8])
@pytest.mark.parametrize("value_range", [["-1", "1"]])
def test_quick(dtype, value_range):
    pass

@deduplicate_bool_ranges
@pytest.mark.parametrize("dtype", [torch.bool])
@pytest.mark.parametrize("unknown", [Unknown()])
@pytest.mark.parametrize("value_range", [["-1", "1"], ["0", "max"]])
def test_unknown(dtype, value_range, unknown):
    pass
""",
    )
    assert selection["kept"] == selection["before"]
