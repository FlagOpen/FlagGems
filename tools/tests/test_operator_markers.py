# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only regressions for generated-operator selection and CI enforcement."""

import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.operator_markers import marker_overrides, operator_marker  # noqa: E402


class OperatorMarkerTests(unittest.TestCase):
    def test_migrated_ids_select_canonical_and_legacy_ids_keep_their_markers(self):
        self.assertEqual(
            operator_marker("_assert_scalar", set()), "underscore_assert_scalar"
        )
        self.assertEqual(operator_marker("_values", {"values"}), "underscore_values")
        self.assertEqual(operator_marker("values", {"values"}), "values")
        self.assertEqual(operator_marker("_reshape_alias", set()), "reshape_alias")
        self.assertEqual(
            operator_marker("_future_op", {"future_op"}), "underscore_future_op"
        )

    def test_entire_migrated_inventory_has_consistent_pairs(self):
        overrides = marker_overrides()
        # Original 297 definitions at 004cfea contain 132 underscore IDs.
        self.assertEqual(len(overrides), 132)
        for op, marker in overrides.items():
            self.assertTrue(op.startswith("_"), op)
            self.assertEqual(marker, "underscore_" + op.lstrip("_"))
            for folder in ("tests", "benchmark"):
                with self.subTest(op=op, folder=folder):
                    tree = ast.parse((ROOT / folder / f"test_{op}.py").read_text())
                    markers = {
                        node.attr
                        for node in ast.walk(tree)
                        if isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Attribute)
                        and node.value.attr == "mark"
                        and isinstance(node.value.value, ast.Name)
                        and node.value.value.id == "pytest"
                    }
                    self.assertIn(marker, markers)
                    self.assertNotIn(op.lstrip("_"), markers)

    def test_cli_derives_same_marker_for_test_benchmark_and_config_changes(self):
        command = [
            sys.executable,
            str(ROOT / "tools/ci_checks/derive_changed_operators.py"),
            "--ops-only",
            "--changed-files",
        ]
        for filename in (
            "tests/test__assert_scalar.py",
            "benchmark/test__assert_scalar.py",
        ):
            with self.subTest(filename=filename):
                result = subprocess.run(
                    command + [filename],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                self.assertEqual(result.stdout, "underscore_assert_scalar\n")
        result = subprocess.run(
            command + ["conf/operator_marker_overrides.json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(
            set(result.stdout.splitlines()), set(marker_overrides().values())
        )

    def test_ci_checks_unregistered_generated_operators_and_both_files(self):
        checker = str(ROOT / "tools/ci_checks/check_operator_markers.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "conf").mkdir()
            (root / "conf/operators.yaml").write_text("ops: []\n")
            for folder in ("tests", "benchmark"):
                (root / folder).mkdir()
            good = "@pytest.mark.underscore_assert_scalar\ndef test_op(): pass\n"
            bad = "@pytest.mark.assert_scalar\ndef test_op(): pass\n"
            files = [
                root / folder / "test__assert_scalar.py"
                for folder in ("tests", "benchmark")
            ]
            for invalid in (
                None,
                "module",
                "class",
                0,
                1,
                "both",
                "syntax",
                "mixed",
                "local",
            ):
                with self.subTest(invalid=invalid):
                    for index, path in enumerate(files):
                        path.write_text(
                            bad if invalid == index or invalid == "both" else good
                        )
                    if invalid == "module":
                        files[0].write_text(
                            "pytestmark = [pytest.mark.underscore_assert_scalar]\ndef test_op(): pass\n"
                        )
                    if invalid == "class":
                        files[0].write_text(
                            "@pytest.mark.underscore_assert_scalar\nclass TestOp:\n    def test_op(self): pass\n"
                        )
                    if invalid == "local":
                        files[0].write_text(
                            "def test_op():\n    pytestmark = pytest.mark.underscore_assert_scalar\n"
                        )
                    if invalid == "syntax":
                        files[1].write_text(good + "def broken(\n")
                    if invalid == "mixed":
                        files[1].write_text(good + bad.replace("test_op", "test_old"))
                    result = subprocess.run(
                        [
                            sys.executable,
                            checker,
                            "--operators",
                            json.dumps(["_assert_scalar"]),
                        ],
                        cwd=root,
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(
                        result.returncode,
                        0 if invalid in (None, "module", "class") else 1,
                        result.stdout + result.stderr,
                    )


if __name__ == "__main__":
    unittest.main()
