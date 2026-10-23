# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit correctness cases, with an opt-in reference-as-candidate mode.

Each operation receives one independently cloned input bundle. Return all
observable outputs (including mutated buffers) so the original comparator can
check the complete contract. This module does not change operator registration.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

_USE_INPUTS = object()
_MODE = ContextVar("flaggems_reference_test_mode", default=None)


class ReferenceContractError(ValueError):
    """A selected test does not expose the explicit reference contract."""


@dataclass
class ReferenceCase:
    inputs: Any
    candidate: Callable
    reference: Callable
    compare: Callable
    reference_inputs: Any = _USE_INPUTS
    clone: Callable = deepcopy

    def __post_init__(self):
        for name in ("reference", "candidate", "compare", "clone"):
            if not callable(getattr(self, name)):
                raise ReferenceContractError(f"ReferenceCase.{name} must be callable")


@dataclass
class _Execution:
    ref_as_candidate: bool
    nodeid: str
    reference_calls: int = 0
    candidate_calls: int = 0


@contextmanager
def reference_test_mode(enabled, nodeid=""):
    """Scope pytest configuration to one test, including failure cleanup."""
    state = _Execution(enabled, nodeid)
    token = _MODE.set(state)
    try:
        yield state
    finally:
        _MODE.reset(token)


def is_reference_test(function):
    return getattr(function, "_flag_gems_reference_test", False) is True


def reference_test(factory):
    """Execute a parametrized factory returning one :class:`ReferenceCase`.

    Normal execution compares reference against candidate. With pytest's
    --ref-as-candidate, both calls use reference on separate input snapshots.
    Keep the candidate callable lazy, e.g. lambda args: flag_gems.add(*args),
    so ordinary CLI overrides still resolve at execution time.
    """

    @wraps(factory)
    def run(*args, **kwargs):
        state = _MODE.get() or _Execution(False, factory.__qualname__)
        case = factory(*args, **kwargs)
        if not isinstance(case, ReferenceCase):
            raise ReferenceContractError(
                f"{state.nodeid}: @reference_test must return ReferenceCase "
                "with explicit inputs, candidate, reference and compare"
            )
        # Validate again in case a factory changed a field after construction.
        case.__post_init__()
        reference_inputs = (
            case.inputs
            if case.reference_inputs is _USE_INPUTS
            else case.reference_inputs
        )
        expected_inputs = case.clone(reference_inputs)
        actual_inputs = case.clone(case.inputs)
        state.reference_calls += 1
        expected = case.reference(expected_inputs)
        operation = case.reference if state.ref_as_candidate else case.candidate
        state.candidate_calls += 1
        if state.ref_as_candidate:
            state.reference_calls += 1
        actual = operation(actual_inputs)
        case.compare(actual, expected)

    run._flag_gems_reference_test = True
    return run


def validate_ref_as_candidate_options(config):
    """Validate only the new mode; leave all existing combinations unchanged."""
    if not getattr(config.option, "ref_as_candidate", False):
        return False
    import pytest

    conflicts = [
        name
        for name in (
            "override",
            "override_config",
            "reference_only",
            "preflight_only",
            "profile_only",
            "list_cases",
            "query",
            "parallel",
            "collect_marks",
        )
        if getattr(config.option, name, None)
    ]
    if conflicts:
        raise pytest.UsageError(
            "--ref-as-candidate cannot be combined with "
            + ", ".join("--" + name.replace("_", "-") for name in conflicts)
        )
    return True
