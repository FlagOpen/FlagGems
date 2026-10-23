# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python contract and real pytest-hook regressions (no accelerator needed)."""

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/flag_gems/testing/reference.py"
SPEC = importlib.util.spec_from_file_location("_flaggems_reference_contract", SOURCE)
reference = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference
SPEC.loader.exec_module(reference)
ReferenceCase = reference.ReferenceCase


def compare(actual, expected):
    assert actual == expected


def forbidden(*args, **kwargs):
    raise AssertionError("original candidate was called")


def test_default_execution_still_calls_candidate():
    @reference.reference_test
    def case():
        return ReferenceCase([1, 2], forbidden, sum, compare)

    with pytest.raises(AssertionError, match="original candidate"):
        case()
    with reference.reference_test_mode(True) as state:
        case()
    assert (state.reference_calls, state.candidate_calls) == (2, 1)


@pytest.mark.parametrize("enabled", [False, True])
def test_independent_inputs_preserve_internal_aliases_and_mutations(enabled):
    original = [1]
    inputs = (original, original)

    def mutate(args):
        assert args[0] is args[1]
        args[0].append(2)
        return args

    @reference.reference_test
    def case():
        return ReferenceCase(inputs, mutate, mutate, compare)

    with reference.reference_test_mode(enabled):
        case()
    assert original == [1]


def test_separate_reference_inputs_and_custom_clone():
    snapshots = []

    def clone(value):
        snapshots.append(tuple(value))
        return list(value)

    @reference.reference_test
    def case(value):
        return ReferenceCase(
            [value],
            forbidden,
            sum,
            compare,
            reference_inputs=[float(value)],
            clone=clone,
        )

    assert list(inspect.signature(case).parameters) == ["value"]
    with reference.reference_test_mode(True):
        case(2)
    assert snapshots == [(2.0,), (2,)]


def test_mode_restores_after_failure_and_nested_execution():
    @reference.reference_test
    def case():
        return ReferenceCase([1], forbidden, sum, compare)

    with reference.reference_test_mode(True):
        with reference.reference_test_mode(False):
            with pytest.raises(AssertionError):
                case()
        case()
    with pytest.raises(AssertionError):
        case()


@pytest.mark.parametrize("field", ["reference", "candidate", "compare", "clone"])
def test_invalid_callable_is_contract_error(field):
    values = dict(inputs=[1], candidate=sum, reference=sum, compare=compare)
    values[field] = None
    with pytest.raises(reference.ReferenceContractError, match=field):
        ReferenceCase(**values)


@pytest.mark.parametrize("value", [None, {}, sum])
def test_factory_must_expose_case(value):
    @reference.reference_test
    def case():
        return value

    with pytest.raises(reference.ReferenceContractError, match="return ReferenceCase"):
        case()


def test_comparison_failure_is_not_swallowed():
    @reference.reference_test
    def case():
        return ReferenceCase([1], lambda _: 100, sum, compare)

    with pytest.raises(AssertionError):
        case()


@pytest.mark.parametrize(
    "option",
    [
        "override",
        "override_config",
        "reference_only",
        "preflight_only",
        "profile_only",
        "list_cases",
        "query",
        "parallel",
        "collect_marks",
    ],
)
def test_new_mode_conflicts_do_not_change_legacy_options(option):
    options = SimpleNamespace(ref_as_candidate=False, **{option: True})
    config = SimpleNamespace(option=options)
    assert reference.validate_ref_as_candidate_options(config) is False
    options.ref_as_candidate = True
    with pytest.raises(pytest.UsageError, match="cannot be combined"):
        reference.validate_ref_as_candidate_options(config)


@pytest.fixture
def project(pytester):
    # Load the actual correctness conftest and override code, but a host-only
    # package shell so these subprocess checks also run without Torch/Triton.
    preamble = f"""
import sys, types
from pathlib import Path
pkg = types.ModuleType("flag_gems")
pkg.__path__ = [{str(ROOT / "src/flag_gems")!r}]
pkg.device = "cpu"
pkg.add = lambda a, b: -100
sys.modules["flag_gems"] = pkg
testing = types.ModuleType("flag_gems.testing")
testing.__path__ = [{str(ROOT / "src/flag_gems/testing")!r}]
sys.modules["flag_gems.testing"] = testing
"""
    pytester.makeconftest(preamble + (ROOT / "tests/conftest.py").read_text())
    pytester.makeini("[pytest]\nmarkers = good\n")
    pytester.makepyfile(test_case="""
import pytest
import flag_gems
from flag_gems.testing.reference import ReferenceCase, reference_test

def compare(actual, expected):
    assert actual == expected

@pytest.mark.good
@pytest.mark.parametrize("value", [1, 2, 3])
@reference_test
def test_case(value):
    def reference(args):
        return sum(args)
    return ReferenceCase((value, 2), lambda args: flag_gems.add(*args), reference, compare)
""")
    return pytester


def test_real_hooks_ref_mode_full_grid_and_call_report(project):
    result = project.runpytest_subprocess("--ref-as-candidate", "--ref", "cpu", "-q")
    result.assert_outcomes(passed=3)
    records = json.loads((project.path / "accuracy_result.json").read_text())
    assert len(records) == 3
    assert all(r["candidate_source"] == "reference" for r in records.values())
    assert all(r["reference_calls"] == 2 for r in records.values())
    assert all(r["reference_candidate_calls"] == 1 for r in records.values())


def test_real_hooks_default_and_external_overrides(project):
    project.runpytest_subprocess("-q").assert_outcomes(failed=3)
    candidate = project.makepyfile(candidate="def run(a, b): return a + b\n")
    project.runpytest_subprocess(
        "-q", "--override", f"add:{candidate}:run"
    ).assert_outcomes(passed=3)
    for suffix in ("json", "yaml"):
        config = project.path / f"override.{suffix}"
        config.write_text(
            json.dumps(
                {"overrides": {"add": {"file": str(candidate), "function": "run"}}}
            )
        )
        project.runpytest_subprocess(
            "-q", "--override-config", str(config)
        ).assert_outcomes(passed=3)


def test_missing_declaration_fails_only_when_enabled_and_selected(project):
    project.makepyfile(test_legacy="def test_legacy(): pass\n")
    project.runpytest_subprocess("-q", "test_legacy.py").assert_outcomes(passed=1)
    result = project.runpytest_subprocess("-q", "--ref-as-candidate")
    result.assert_outcomes(passed=3, errors=1)
    result.stdout.fnmatch_lines(
        ["*test_legacy.py::test_legacy*requires @reference_test*"]
    )
    project.runpytest_subprocess(
        "-q", "--ref-as-candidate", "-m", "good"
    ).assert_outcomes(passed=3, deselected=1)


def test_bad_factory_is_not_accepted_as_ref_validation(project):
    project.makepyfile(test_bad="""
from flag_gems.testing.reference import reference_test
@reference_test
def test_bad():
    reference = lambda args: args
""")
    result = project.runpytest_subprocess("-q", "--ref-as-candidate", "test_bad.py")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*must return ReferenceCase*"])


def test_declaration_without_execution_cannot_pass(project):
    project.makepyfile(test_empty="""
def test_empty():
    pass
test_empty._flag_gems_reference_test = True
""")
    result = project.runpytest_subprocess("-q", "--ref-as-candidate", "test_empty.py")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*did not execute a candidate call*"])
