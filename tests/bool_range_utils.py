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

"""Opt-in collection of one equivalent bool value range per workload."""

import torch


def deduplicate_bool_ranges(function=None, *, dtype_arg="dtype"):
    """Declare that value_range only changes bool sampling, not the workload.

    Apply only after checking the entire fixture: value_range must not change
    another operand's values, structure, expected result or execution branch.
    Other parameter combinations and parameter-specific marks remain distinct.
    The input generator and the retained pytest items are left untouched.
    """

    def decorate(test):
        test._bool_range_dtype_arg = dtype_arg
        return test

    return decorate if function is None else decorate(function)


def _parameter_key(value):
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_parameter_key(item) for item in value))
    if isinstance(value, dict):
        return (
            dict,
            frozenset((key, _parameter_key(item)) for key, item in value.items()),
        )
    hash(value)
    return (type(value), value)


def deduplicate_bool_range_items(items):
    """Return retained and deselected items, preserving their original order."""
    seen = set()
    retained = []
    deselected = []
    for item in items:
        dtype_arg = getattr(getattr(item, "obj", None), "_bool_range_dtype_arg", None)
        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec is not None else {}
        if (
            not dtype_arg
            or params.get(dtype_arg) is not torch.bool
            or "value_range" not in params
        ):
            retained.append(item)
            continue
        try:
            key = (
                item.parent.nodeid,
                item.originalname,
                frozenset(
                    (name, _parameter_key(value))
                    for name, value in params.items()
                    if name != "value_range"
                ),
                tuple(
                    (mark.name, _parameter_key(mark.args), _parameter_key(mark.kwargs))
                    for mark in item.iter_markers()
                    if mark.name != "parametrize"
                ),
            )
            hash(key)
        except TypeError:
            # An unfamiliar unhashable parameter cannot establish equivalence.
            retained.append(item)
            continue
        if key in seen:
            deselected.append(item)
        else:
            seen.add(key)
            retained.append(item)
    return retained, deselected
