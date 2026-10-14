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

"""Benchmark-only switch back to the pinned implementation without warmup."""

import ast
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path


def pytest_configure(config):
    # Remove only the added entry call, preserving function identity so aliases
    # already imported by flag_gems still execute the exact original body.
    for module_name, names in (
        ("embedding_bag", ("_embedding_bag", "_embedding_bag_forward_only")),
        ("embedding_bag_backward", ("_embedding_bag_backward",)),
    ):
        module = importlib.import_module(
            f"flag_gems.runtime.backend._iluvatar.ops.{module_name}"
        )
        for name in names:
            function = getattr(module, name)
            tree = ast.parse(inspect.getsource(function))
            definition = tree.body[0]
            removed = [
                node
                for node in definition.body
                if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "_prepare_native_assert"
            ]
            assert len(removed) == 1
            definition.body.remove(removed[0])
            namespace = {}
            exec(
                compile(tree, inspect.getfile(function), "exec"),
                module.__dict__,
                namespace,
            )
            function.__code__ = namespace[name].__code__

    # Restore the original constexpr checker as well as removing warmup calls.
    # This snapshot is exported from the pinned commit before applying the patch.
    source = Path.cwd().parent / "ix-results" / "baseline_embedding_bag.py"
    spec = importlib.util.spec_from_file_location("ix_pinned_embedding_bag", source)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    forward = importlib.import_module(
        "flag_gems.runtime.backend._iluvatar.ops.embedding_bag"
    )
    forward._embedding_bag_check_flags_native = (
        original._embedding_bag_check_flags_native
    )
    forward._check_native_error.__code__ = original._check_native_error.__code__
