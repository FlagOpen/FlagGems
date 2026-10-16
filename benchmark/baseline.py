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

"""Optional ops-baseline matching; no tensor values or device imports needed."""

import json
import math
import warnings


def positive_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def describe_call(args, kwargs):
    """Describe inputs before either implementation can mutate them."""
    import torch

    def describe(value):
        if isinstance(value, torch.Tensor):
            result = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if value.layout == torch.strided:
                result["stride"] = list(value.stride())
            result["requires_grad"] = value.requires_grad
            return result
        if isinstance(value, (list, tuple)):
            return [describe(item) for item in value]
        if isinstance(value, dict):
            return {key: describe(item) for key, item in value.items()}
        if isinstance(value, torch.dtype):
            return str(value)
        return value

    return {"args": describe(args), "kwargs": describe(kwargs)}


def _matches(expected, actual):
    if isinstance(expected, dict):
        if "scalar" in expected:
            return _matches(expected["scalar"], actual)
        if not isinstance(actual, dict):
            return False
        if "shape" in expected:
            # Ignore source-only tensor annotations, but compare every available
            # identity field, including stride and dtype for auxiliary tensors.
            return all(
                key not in expected or _matches(expected[key], actual.get(key))
                for key in ("shape", "dtype", "stride", "requires_grad")
            )
        return set(expected) == set(actual) and all(
            _matches(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(expected) == len(actual)
            and all(_matches(a, b) for a, b in zip(expected, actual))
        )
    return type(expected) is type(actual) and expected == actual


def _tensors(value):
    if isinstance(value, dict):
        if "shape" in value:
            yield value
        else:
            for item in value.values():
                yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _legacy_shape_key(key, call):
    """Legacy records have no config, so only an exact simple shape key works."""
    first = next(_tensors(call), None)
    if first is None:
        return False
    try:
        return json.loads(key) == first["shape"]
    except (TypeError, ValueError):
        return False


class BaselineData:
    def __init__(self, data, source):
        self.data = data
        self.source = str(source)
        self._records = {}

    @classmethod
    def load(cls, path):
        if not path:
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("top-level baseline must be a JSON object")
        except (OSError, ValueError) as exc:
            warnings.warn(
                f"Cannot load --base-data {path}: {exc}; using native baseline"
            )
            return None
        return cls(data, path)

    def lookup(self, op_name, dtype, call, named_call=None):
        """Return (record, shape key, miss reason), never guess an ambiguous match.

        Supports ops-baseline PyTorch args/kwargs metadata, named SGLang/vLLM
        config.inputs when parameter binding is available, and legacy simple
        shape keys without input metadata. Leading-underscore ops are valid.
        """
        entry = self.data.get(op_name)
        if not isinstance(entry, dict):
            return None, None, "operator_missing"
        shapes = entry.get("shapes")
        if not isinstance(shapes, dict):
            return None, None, "shape_missing"
        index_key = (op_name, str(dtype))
        if index_key not in self._records:
            self._records[index_key] = [
                (key, per_dtype[str(dtype)])
                for key, per_dtype in shapes.items()
                if isinstance(per_dtype, dict)
                and isinstance(per_dtype.get(str(dtype)), dict)
            ]
        candidates = []
        for key, record in self._records[index_key]:
            config = record.get("config")
            if config is not None and not isinstance(config, dict):
                continue
            inputs = config.get("inputs") if isinstance(config, dict) else None
            if (
                isinstance(config, dict)
                and "inputs" in config
                and not isinstance(inputs, dict)
            ):
                continue
            if isinstance(inputs, dict):
                if "args" in inputs or "kwargs" in inputs:
                    matched = _matches(inputs, call)
                else:
                    matched = named_call is not None and all(
                        name in named_call and _matches(value, named_call[name])
                        for name, value in inputs.items()
                    )
            else:
                matched = _legacy_shape_key(key, call)
            if matched:
                candidates.append((record, key))
        if len(candidates) != 1:
            reason = "ambiguous_match" if candidates else "shape_or_dtype_missing"
            return None, None, reason
        record, key = candidates[0]
        latency = record.get("latency_ms")
        if latency is None and positive_number(record.get("T_us")):
            latency = record["T_us"] / 1000
        if not positive_number(latency):
            return None, None, "invalid_latency"
        return {**record, "latency_ms": latency}, key, None

    def scaling_factor(self, vendor, chip, unit, dtype):
        scaling = self.data.get("_scaling_factors")
        if not isinstance(scaling, dict) or not isinstance(scaling.get("chips"), list):
            return None
        candidates = [
            c
            for c in scaling["chips"]
            if isinstance(c, dict) and c.get("vendor") == vendor
        ]
        matched = [
            c
            for c in candidates
            if isinstance(c.get("chip"), str)
            and chip
            and c["chip"].lower() in chip.lower()
        ]
        if len(matched) == 1:
            selected = matched[0]
        elif not chip and len(candidates) == 1:
            selected = candidates[0]
        else:
            return None
        if not isinstance(unit, str):
            return None
        resource = {"mem": "bandwidth_gbps", "cuda": "vector", "tensor": "tensor"}.get(
            unit
        )
        dtype_key = {
            "torch.float16": "fp16",
            "torch.bfloat16": "bf16",
            "torch.float32": "fp32",
            "torch.float64": "fp64",
            "torch.int8": "int8",
            "torch.float8_e4m3fn": "fp8",
            "torch.float8_e5m2": "fp8",
        }.get(str(dtype))
        if resource is None:
            return None
        for preference in ("measured", "nominal"):
            block = selected.get(preference)
            if not isinstance(block, dict):
                continue
            value = block.get(resource)
            if resource != "bandwidth_gbps":
                value = value.get(dtype_key) if isinstance(value, dict) else None
            if positive_number(value):
                return value
        return None

    def apply(self, metric, op_name, dtype, call, named_call, vendor, chip):
        if not positive_number(metric.latency):
            metric.base_miss_reason = "invalid_candidate_latency"
            return
        record, key, reason = self.lookup(op_name, dtype, call, named_call)
        metric.base_miss_reason = reason
        if record is None:
            return
        factor = self.scaling_factor(
            vendor, chip, record.get("bottle_neck_unit"), dtype
        )
        raw_ratio = record["latency_ms"] / metric.latency
        compared = raw_ratio / (factor or 1.0)
        if not positive_number(compared):
            metric.base_miss_reason = "invalid_speedup"
            return
        metric.compared_speedup = compared
        metric.base_raw_speedup = raw_ratio
        metric.base_data_source = self.source
        metric.base_shape_key = key
        metric.base_latency_ms = record["latency_ms"]
        metric.base_scaling_factor = factor
