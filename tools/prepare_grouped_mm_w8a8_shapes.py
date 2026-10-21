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

"""Convert external SGLang grouped INT8 captures for the standard benchmark.

Input: shapes.json plus metadata/<model>-<scenario>.jsonl, as exported in the
FlagOSTune capture bundle. Output stays outside the repository, for example:

python tools/prepare_grouped_mm_w8a8_shapes.py \
    --input-dir /data/real_shapes/grouped_gemm_nt_i8i8bf16_nopad \
    --output /tmp/grouped_mm_w8a8.json
python -m pytest benchmark/test_grouped_mm_w8a8.py --dtypes int8 \
    --shape_file /tmp/grouped_mm_w8a8.json --list-cases

Keep every scenario's first-observed expert counts and call weight separately.
Never substitute a synthetic partition for missing routing metadata. Tensor
contents are not captured; the benchmark generates deterministic values/scales.
This converter needs only the Python standard library, not Torch or a GPU.
"""

import argparse
import hashlib
import json
from itertools import accumulate
from pathlib import Path

OPERATOR = "torch.ops.sglang.grouped_gemm_nt_i8i8bf16_nopad"


def positive_int(value):
    return type(value) is int and value > 0


def validate_metadata(record, geometry):
    m, n, k, g = geometry
    lengths = record["m_rows_first_observation"]
    if (
        len(lengths) != g
        or any(type(v) is not int or v < 0 for v in lengths)
        or sum(lengths) != m
    ):
        raise ValueError(f"Invalid captured expert counts for {geometry}")
    tensors = record["tensors"]
    expected = {
        "A": ([m, k], "torch.int8"),
        "As": ([m, 1], "torch.float32"),
        "B": ([g, n, k], "torch.int8"),
        "Bs": ([g, n, 1], "torch.float32"),
        "C": ([m, n], "torch.bfloat16"),
        "m_indices": ([m], "torch.int32"),
        "m_rows": ([g], "torch.int32"),
    }
    for name, (shape, dtype) in expected.items():
        tensor = tensors[name]
        if tensor["shape"] != shape or tensor["dtype"] != dtype:
            raise ValueError(f"Unsupported {name} shape/dtype for {geometry}")
        strides = tensor["stride"]
        if len(strides) != len(shape):
            raise ValueError(f"Invalid {name} strides for {geometry}")
        stride = 1
        for size, actual in zip(reversed(shape), reversed(strides)):
            if size > 1 and actual != stride:
                raise ValueError(f"Noncontiguous {name} for {geometry}")
            stride *= size
    return lengths


def convert(input_dir):
    input_dir = Path(input_dir)
    source = (input_dir / "shapes.json").read_bytes()
    exports = json.loads(source)
    hashes = {"shapes.json": hashlib.sha256(source).hexdigest()}
    metadata = {}
    rows = []
    seen = set()
    for source_index, entry in enumerate(exports):
        if entry["op"] != OPERATOR:
            continue
        geometry = entry["shape"]
        if (
            len(geometry) != 4
            or not all(positive_int(v) for v in geometry)
            or entry["labels"] != "M,N,K,E"
            or entry["extra"] != "(int8,int8,bfloat16)"
        ):
            raise ValueError(f"Unsupported capture row {source_index}")
        m, n, k, g = geometry
        if k % 128 or k > 131071:
            raise ValueError(f"Unsupported INT8 baseline K={k}")
        sources = entry["sources"]
        if (
            not sources
            or not all(positive_int(v) for v in sources.values())
            or sum(sources.values()) != entry["count"]
        ):
            raise ValueError(f"Invalid source call counts in row {source_index}")
        for source_name, count in sources.items():
            parts = source_name.split("/")
            if len(parts) != 2 or any(p in ("", ".", "..") for p in parts):
                raise ValueError(f"Expected model/scenario source: {source_name}")
            scenario = parts[1]
            key = (source_name, tuple(geometry))
            if key in seen:
                raise ValueError(f"Duplicate scenario/shape: {key}")
            seen.add(key)
            filename = "metadata/" + "-".join(parts) + ".jsonl"
            if filename not in metadata:
                raw = (input_dir / filename).read_bytes()
                hashes[filename] = hashlib.sha256(raw).hexdigest()
                records = {}
                for line in raw.splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    dims = tuple(record["dimensions"][d] for d in ("M", "N", "K", "E"))
                    record_key = (record["scenario"], dims)
                    if record_key in records:
                        raise ValueError(
                            f"Ambiguous metadata in {filename}: {record_key}"
                        )
                    records[record_key] = record
                metadata[filename] = records
            record_key = (scenario, tuple(geometry))
            if record_key not in metadata[filename]:
                raise ValueError(f"Missing captured routing: {source_name}, {geometry}")
            record = metadata[filename][record_key]
            lengths = validate_metadata(record, geometry)
            rows.append(
                {
                    "index": len(rows),
                    "source_index": source_index,
                    "stage": "gate_up" if n < k else "down",
                    "source_shape": geometry,
                    "source_count": count,
                    "source_sources": {source_name: count},
                    "G": g,
                    "M": m,
                    "N": n,
                    "K": k,
                    "lengths": lengths,
                    "offs": list(accumulate(lengths)),
                    "routing_seed": None,
                    "scenario": scenario,
                    "tensor_metadata": record["tensors"],
                }
            )
    if not rows:
        raise ValueError(f"No {OPERATOR} records in shapes.json")
    return {
        "schema_version": 1,
        "reference": "mctlass",
        "input_dtype": "int8",
        "classification": "captured_operator_shapes_with_first_observed_expert_counts",
        "source_sha256": hashes["shapes.json"],
        "source_files_sha256": hashes,
        "source_operator": OPERATOR,
        "unique_geometry_count": len({tuple(row["source_shape"]) for row in rows}),
        "scenario_case_count": len(rows),
        "source_weight_total": sum(row["source_count"] for row in rows),
        "routing_origin": "captured first m_rows for each scenario and geometry; "
        "grouped m_indices reconstructed from counts",
        "value_origin": "synthetic deterministic INT8 operands and FP32 row/column scales",
        "stage_origin": "projection label inferred from N < K (gate_up), else down",
        "sglang_metax_call_path_verified": False,
        "shapes": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        dataset = convert(args.input_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Invalid capture: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2) + "\n")
    print(
        f"Wrote {len(dataset['shapes'])} scenario records, "
        f"{dataset['unique_geometry_count']} geometries, "
        f"{dataset['source_weight_total']} calls to {args.output}"
    )


if __name__ == "__main__":
    main()
