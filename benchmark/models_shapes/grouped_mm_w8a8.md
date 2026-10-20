# grouped_mm_w8a8 benchmark data

`grouped_mm_w8a8.json` preserves the diagnostic dataset used on MetaX C550.
It contains 122 captured `fused_moe` outer shapes, each mapped to gate/up and
down GEMMs (244 cases per dtype). Both benchmark levels run the complete list.
The source export was `/data/fzy/real_shapes_20261012/group_mm_w8a8_w8a16/shapes.json`;
its SHA256 and per-model/scenario counts are embedded in the JSON.

**This is not a captured grouped-MM invocation trace.** Expert routing and its
cumulative offsets are synthetic, generated once and stored in the dataset.
Operands and per-row/per-column FP32 scales are regenerated deterministically.
The mapping assumes all 256 experts are local, `M = tokens * top_k`, and SwiGLU
splits the gate/up width in half. Router, packing, activation, scatter and
reduction costs are excluded. `source_count` counts outer MoE calls: gate/up
and down share that count, not two independent model calls.

The standard benchmark calls `flag_gems.grouped_mm_w8a8` directly and uses
`mctlassEx.ContinuousGroupedGEMM.ContinuousGroupedGEMM` as the reference.
Both receive identical operands/scales/partitions and return BF16. The baseline
includes output allocation and tile-prefix construction on every call. INT8
uses the native metadata helper. FP8 uses a Torch GPU ceil-div/cumsum/cat prefix
adapter because the native FP8 metadata helper failed sparse-routing validation;
row/column scales are repeated over K/128 blocks inside the measured call.
FP8 results therefore compare against **adapted mctlass**, not its unmodified
metadata helper. Native block-size selection and input routing metadata are
prepared outside timing. Missing mctlass bindings cause a skip, never a fallback
reference implementation.

From the repository root in the MetaX runtime environment:

```bash
# Shared accuracy suite (no performance assertions).
python -m pytest tests/test_scaled_grouped_mm.py -q --ref cpu

# Tensor-free case manifest, including exact offsets and source counts.
python -m pytest benchmark/test_grouped_mm_w8a8.py -q --level core \
  --list-cases --record json --output grouped-mm-cases.json

# Standard Event median timing, both INT8 and FP8, all 488 cases.
python -m pytest benchmark/test_grouped_mm_w8a8.py -q --level core \
  --mode kernel --warmup 20 --iter 100 \
  --record json --output grouped-mm-performance.json

# Optional dtype filter: --dtypes int8 or --dtypes float8_e4m3fn.
# Optional single case (copy the exact ID from the manifest):
python -m pytest benchmark/test_grouped_mm_w8a8.py -q --level core \
  --mode kernel --warmup 20 --iter 100 \
  --case-id 'benchmark/test_grouped_mm_w8a8.py::test_grouped_mm_w8a8::core::int8::106' \
  --record json --output grouped-mm-single.json
```

A replacement export can be supplied with `--shape_file path/to/export.json`
using the same schema. An official target trace must contain its actual
partitions and scale contract; outer dimensions alone cannot establish these.
For the diagnostic aggregate, join result `case_id` to the manifest and compute
`sum(source_count * latency_base) / sum(source_count * latency)` per dtype.
This is a ratio of weighted GEMM times, not an end-to-end model speedup and not
the arithmetic mean of per-shape speedups. Do not compare these kernel-mode
measurements directly to CUDA Graph, wrapper or operator-mode timings.
