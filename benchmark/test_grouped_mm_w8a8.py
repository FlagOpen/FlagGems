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

"""Standard benchmark for MetaX grouped_mm_w8a8 against mctlass Torch bindings.

The bundled export contains 122 real fused_moe outer shapes, expanded into 244
GEMMs. Expert routing/offsets, operands and row/column scales are synthetic:
these are diagnostic workloads, not native grouped-MM model traces. Both core
and comprehensive cover the entire export. --shape_file accepts the same JSON
schema. --list-cases records the source count, offsets and provenance without
allocating device tensors; --case-id selects individual cases.

Baseline includes output allocation, tile-prefix generation and FP8 scale
replication on every call. FP8 uses the validated Torch tile-prefix adapter,
not the native FP8 metadata helper (which failed sparse-routing validation).
Native block selection and supplied routing metadata are prepared outside
measurement. Timing, warmup, repetition and JSON output belong to base.Benchmark.
Accuracy tests belong to tests/test_scaled_grouped_mm.py.
"""

import json
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base, consts


class GroupedMMW8A8Benchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        path = Path(shape_file_path or base.Config.shape_file)
        default = Path(__file__).with_name("core_shapes.yaml")
        if path.resolve() == default.resolve():
            path = Path(__file__).parent / "models_shapes" / "grouped_mm_w8a8.json"
        self.dataset = json.loads(path.read_text())
        self.shapes = self.dataset["shapes"]
        self.shape_desc = "G, M, N, K; real MoE outer dimensions, synthetic routing"
        for row in self.shapes:
            lengths = row["lengths"]
            ends, total = [], 0
            for length in lengths:
                if not isinstance(length, int) or length < 0:
                    raise ValueError("Expert lengths must be nonnegative integers")
                total += length
                ends.append(total)
            if (
                len(lengths) != row["G"]
                or total != row["M"]
                or ends != row["offs"]
                or row["source_count"] <= 0
                or row["K"] % 128
            ):
                raise ValueError(f"Invalid grouped-MM export row {row['index']}")

    def get_case_iter(self, dtype):
        for ordinal, row in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                consts.BenchmarkCasePlan(
                    shape={key: row[key] for key in ("G", "M", "N", "K")},
                    params={
                        key: row[key]
                        for key in (
                            "index",
                            "source_index",
                            "stage",
                            "source_shape",
                            "source_count",
                            "source_sources",
                            "offs",
                            "routing_seed",
                        )
                    }
                    | {
                        "classification": self.dataset["classification"],
                        "source_sha256": self.dataset["source_sha256"],
                        "scale_layout": "FP32 per-row A[M], per-column B[G,N]",
                        "baseline": "mctlass Torch; FP8 uses Torch tile-prefix adapter",
                    },
                    builder_args=(row,),
                ),
            )

    def build_inputs(self, case):
        row = case.builder_args[0].builder_args[0]
        g, m, n, k = (row[key] for key in ("G", "M", "N", "K"))
        dtype = case.dtype
        generator = torch.Generator().manual_seed(
            20261014 + row["index"] + (10000 if dtype == torch.int8 else 0)
        )
        if dtype == torch.int8:
            a = torch.randint(-128, 128, (m, k), dtype=dtype, generator=generator)
            b = torch.randint(-128, 128, (g, n, k), dtype=dtype, generator=generator)
            sa = torch.tensor([2.0 ** (i % 4 - 8) for i in range(m)])
            sb = torch.tensor([2.0 ** (i % 5 - 8) for i in range(g * n)])
        else:
            a = torch.empty((m, k), dtype=dtype)
            for start in range(0, m, 4096):
                end = min(start + 4096, m)
                a[start:end] = (
                    torch.randn((end - start, k), generator=generator) * 0.25
                ).to(dtype)
            b = torch.empty((g, n, k), dtype=dtype)
            for group in range(g):
                b[group] = (torch.randn((n, k), generator=generator) * 0.25).to(dtype)
            sa = torch.tensor([2.0 ** (i % 4 - 1) for i in range(m)])
            sb = torch.tensor([2.0 ** (i % 3 - 1) for i in range(g * n)])
        a, b = a.cuda(), b.cuda()
        sa, sb = sa.float().cuda(), sb.float().reshape(g, n).cuda()
        offsets = torch.tensor(row["offs"], dtype=torch.int32, device="cuda")
        self.segments = torch.tensor(
            [0] + row["offs"], dtype=torch.int32, device="cuda"
        )
        self.weights = torch.arange(g, dtype=torch.int32, device="cuda")
        self.native_config = {"n_group": g}
        native_sa, native_sb = sa, sb
        if dtype != torch.int8:
            self.native_config.update(
                is_blockwise=True,
                use_fp8=True,
                is_scale_a_1d=True,
                is_scale_b_1d=True,
                scale_a_layout="k-major",
                scale_b_layout="k-major",
                M=m,
                m_indices=torch.tensor(
                    [
                        group
                        for group, length in enumerate(row["lengths"])
                        for _ in range(length)
                    ],
                    dtype=torch.int32,
                    device="cuda",
                ),
            )
            native_sa, native_sb = self._fp8_scales(sa, sb, k)
        native_module = pytest.importorskip("mctlassEx.ContinuousGroupedGEMM")
        self.native = native_module.ContinuousGroupedGEMM()
        scratch = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        self.block_m = self.native.get_blocksize_m(
            a, b, scratch, native_sa, native_sb, **self.native_config
        )
        if not isinstance(self.block_m, int) or self.block_m <= 0:
            raise RuntimeError(f"Invalid mctlass block size: {self.block_m}")
        return (
            a,
            b.transpose(-1, -2),
            sa,
            sb,
            {"offs": offsets, "out_dtype": torch.bfloat16},
        )

    @staticmethod
    def _fp8_scales(sa, sb, k):
        sa, sb = sa[:, None], sb[:, :, None]
        if k > 128:
            sa, sb = sa.repeat(1, k // 128), sb.repeat(1, 1, k // 128)
        return sa, sb

    def mctlass_reference(self, a, b, sa, sb, *, offs, out_dtype):
        # All input tensors match the candidate; only the weight view differs.
        m, k = a.shape
        g, _, n = b.shape
        b = b.transpose(-1, -2)
        output = torch.empty((m, n), dtype=out_dtype, device=a.device)
        if a.dtype == torch.int8:
            tiles = torch.empty((g + 1,), dtype=torch.int32, device=a.device)
            self.native.compute_m_num_tiles_indptr(
                a, b, output, self.segments, tiles, self.block_m, **self.native_config
            )
        else:
            counts = torch.div(
                self.segments[1:] - self.segments[:-1] + self.block_m - 1,
                self.block_m,
                rounding_mode="floor",
            )
            tiles = torch.cat(
                (
                    torch.zeros((1,), dtype=torch.int32, device=a.device),
                    torch.cumsum(counts, dim=0, dtype=torch.int32),
                )
            )
            sa, sb = self._fp8_scales(sa, sb, k)
        self.native(
            [m, n, k],
            a,
            b,
            output,
            sa,
            sb,
            None,
            self.segments,
            self.weights,
            tiles,
            **self.native_config,
        )
        return output


@pytest.mark.grouped_mm_w8a8
@pytest.mark.skipif(
    flag_gems.vendor_name != "metax", reason="Requires MetaX mctlass Torch bindings"
)
def test_grouped_mm_w8a8():
    bench = GroupedMMW8A8Benchmark(
        op_name="grouped_mm_w8a8",
        torch_op=None,
        gems_op=flag_gems.grouped_mm_w8a8,
        dtypes=[torch.int8, torch.float8_e4m3fn],
    )
    bench.torch_op = bench.mctlass_reference
    bench.run()
