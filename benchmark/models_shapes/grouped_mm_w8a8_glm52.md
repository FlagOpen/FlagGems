# Captured GLM-5.2 INT8 grouped GEMM / MetaX mctlass comparison

Use `grouped_mm_w8a8_glm52.json` with the standard repository benchmark:

```bash
python -m pytest benchmark/test_grouped_mm_w8a8.py -q --level core \
  --dtypes int8 --shape_file benchmark/models_shapes/grouped_mm_w8a8_glm52.json \
  --list-cases --record json --output glm52-cases.json
python -m pytest benchmark/test_grouped_mm_w8a8.py -q --level core \
  --dtypes int8 --shape_file benchmark/models_shapes/grouped_mm_w8a8_glm52.json \
  --mode kernel --warmup 20 --iter 100 --record json --output glm52-performance.json
```

Source: `/data/fzy/real_shapes_20261019/grouped_gemm_nt_i8i8bf16_nopad`.
The trace records actual PPU SGLang operator calls from GLM-5.2 W8A8 INT8,
TP=8, rank 0, with graphs disabled. There are 26 distinct `(M,N,K,E)` shapes,
28 scenario/shape records, and 3,240,600 calls. Two scenarios sharing a shape
retain their own first-observed expert counts rather than sharing a synthetic
partition. E=257 includes 256 routed experts and one shared expert.

The JSON contains captured `m_rows` as `lengths`, its cumulative `offs`, tensor
shapes/strides/dtypes and scenario call counts. Only the first expert-count
vector per scenario/shape was recorded; it does not capture routing variation
across all calls. Values and FP32 scales are regenerated deterministically.
Grouped expert IDs (`m_indices`) are reconstructed from those captured counts;
the original index values were not exported. This is packed grouped-GEMM replay,
not a replay of every model tensor or end-to-end MoE execution.

`reference: mctlass` selects the installed
`mctlassEx.ContinuousGroupedGEMM.ContinuousGroupedGEMM` Torch interface.
For signed INT8 A/B and BF16 C, this dispatches to native
`mctlass_contiguous_group_gemm`. It does not benchmark the DeepGEMM Python
wrapper or substitute a PyTorch GEMM. Missing native bindings cause a skip.

Both implementations receive identical INT8 operands, FP32 per-row/per-column
scales, captured expert counts and caller-owned preallocated BF16 C. The timed
baseline invocation includes:

1. A GPU metadata adapter that constructs segment offsets and identity weight
   indices from the captured counts (one Triton kernel; no GEMM work).
2. Native block-size selection and `compute_m_num_tiles_indptr`.
3. The unmodified `ContinuousGroupedGEMM` Torch call and native INT8 GEMM.

Metadata buffers are allocated within the invocation; dynamic metadata is not
cached or copied to the CPU. The candidate public entry receives
`(A, As, B, Bs, C, m_indices, m_rows)`, builds its own device expert prefix on
every call, and writes C directly. There is no temporary GEMM output or copy.
Initial input/output allocation, input generation, JIT compilation and wrapper
object construction are outside timing on both sides. Standard kernel-mode
Event median timing includes the GPU metadata work and any launch gaps within
the measured interval; it is not a separate host wall-clock measurement.
The MetaX entry also accepts matching E4M3 inputs; this dataset is INT8 only.
Accuracy coverage is in `tests/test_grouped_mm_w8a8.py`.

The source capture used PPU SGLang 0.5.13+v0.1.0.ppu2.1.0 and PPU DeepGEMM
1.0.0+v0.2.0.ppu2.1.0. This replays its captured group sizes against the native
MetaX baseline. It does not measure model inference or certify the exact
calling path of a particular MetaX SGLang deployment. Router, token packing,
activation, scatter and reduction are outside this operator's boundary.

Compute the weighted total-time ratio as
`sum(source_count * reference_ms) / sum(source_count * candidate_ms)` over all
28 scenario records. Counts already represent operator calls (including both
projections), not additional inferred model invocations. Preserve repeated
measurements per case rather than treating repetitions as more source calls.

## MetaX C550 measurement, 2026-10-21

Physical GPU 2 (container-visible GPU 0), C550 64 GB; Torch
2.10.0+metax3.8.1.0, Triton/FlagTree 3.6.0. Three complete standard benchmark
runs, Event median with 20 ms warmup / 100 ms repetition; each case below uses
the median of its three run medians. All 84 benchmark measurements completed.
Both implementations passed full-output checks against CPU FP64 followed by
BF16 conversion on all 28 records; every generated output was bitwise equal.
This accuracy statement applies to these deterministic operands/scales.

| Projection | Records | Call weight | mctlass weighted ms | FlagGems weighted ms | Weighted total-time ratio | Faster cases |
|---|---:|---:|---:|---:|---:|---:|
| all | 28 | 3240600 | 0.181104 | 0.145347 | 1.246x | 14/28 |
| gate_up | 14 | 1620300 | 0.130581 | 0.142925 | 0.914x | 0/14 |
| down | 14 | 1620300 | 0.231628 | 0.147768 | 1.568x | 14/14 |

The call-count-weighted arithmetic mean of per-case speedups is
**1.230x**; this is distinct from the weighted total-time ratio above.
Neither aggregate reaches 1.3x. Every down projection is faster, but every
gate/up projection is slower than native mctlass. The high-frequency small-M
gate/up calls are important to the weighted target; large-M gate/up cases
expose the largest throughput gap. This is an operator comparison with the
metadata boundary described above, not model inference speedup.

Maximum three-run latency span `(max/min - 1)` is
0.54% for mctlass and
2.59% for FlagGems.
Raw JSON, CSV, accuracy records and source hashes are retained on the server in
`/data/ldc/fzy/group-mm-mixed-20261009/mctlass-real-20261021`.

All rows have E=257. Speedup = mctlass / FlagGems; less than 1 means slower.

| # | Scenario | Projection | M | N | K | Count | mctlass ms | FlagGems ms | Speedup |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | p4096d1024 | gate_up | 45 | 512 | 6144 | 921600 | 0.104960 | 0.107520 | 0.976x |
| 1 | p4096d1024 | down | 45 | 6144 | 256 | 921600 | 0.143616 | 0.122112 | 1.176x |
| 2 | p1024d1024 | gate_up | 117 | 512 | 6144 | 441600 | 0.171392 | 0.183808 | 0.932x |
| 3 | p1024d1024 | down | 117 | 6144 | 256 | 441600 | 0.346880 | 0.170240 | 2.038x |
| 4 | p4096d1024 | gate_up | 36 | 512 | 6144 | 76800 | 0.072192 | 0.083200 | 0.868x |
| 5 | p4096d1024 | down | 36 | 6144 | 256 | 76800 | 0.135168 | 0.102912 | 1.313x |
| 6 | p1024d1024 | gate_up | 126 | 512 | 6144 | 62400 | 0.130048 | 0.157952 | 0.823x |
| 7 | p1024d1024 | down | 126 | 6144 | 256 | 62400 | 0.397312 | 0.143872 | 2.762x |
| 8 | p1024d1024 | gate_up | 72 | 512 | 6144 | 62250 | 0.148736 | 0.162304 | 0.916x |
| 9 | p1024d1024 | down | 72 | 6144 | 256 | 62250 | 0.213248 | 0.148480 | 1.436x |
| 10 | p1024d1024 | gate_up | 108 | 512 | 6144 | 43200 | 0.167936 | 0.180736 | 0.929x |
| 11 | p1024d1024 | down | 108 | 6144 | 256 | 43200 | 0.324608 | 0.166656 | 1.948x |
| 12 | p1024d1024 | gate_up | 36864 | 512 | 6144 | 1875 | 1.235456 | 2.336640 | 0.529x |
| 13 | p4096d1024 | gate_up | 36864 | 512 | 6144 | 4800 | 1.105664 | 2.214144 | 0.499x |
| 14 | p1024d1024 | down | 36864 | 6144 | 256 | 1875 | 3.678208 | 2.539904 | 1.448x |
| 15 | p4096d1024 | down | 36864 | 6144 | 256 | 4800 | 3.671936 | 2.495488 | 1.471x |
| 16 | p1024d1024 | gate_up | 90 | 512 | 6144 | 4650 | 0.161024 | 0.175104 | 0.920x |
| 17 | p1024d1024 | down | 90 | 6144 | 256 | 4650 | 0.257536 | 0.161792 | 1.592x |
| 18 | p1024d1024 | gate_up | 9216 | 512 | 6144 | 600 | 0.616704 | 1.535744 | 0.402x |
| 19 | p1024d1024 | down | 9216 | 6144 | 256 | 600 | 1.485568 | 0.717824 | 2.070x |
| 20 | p1024d1024 | gate_up | 81 | 512 | 6144 | 300 | 0.152064 | 0.164864 | 0.922x |
| 21 | p1024d1024 | down | 81 | 6144 | 256 | 300 | 0.244992 | 0.149760 | 1.636x |
| 22 | p1024d1024 | gate_up | 16704 | 512 | 6144 | 75 | 0.833024 | 1.421056 | 0.586x |
| 23 | p1024d1024 | down | 16704 | 6144 | 256 | 75 | 1.875200 | 1.668608 | 1.124x |
| 24 | p1024d1024 | gate_up | 17856 | 512 | 6144 | 75 | 0.774272 | 1.365248 | 0.567x |
| 25 | p1024d1024 | down | 17856 | 6144 | 256 | 75 | 2.747904 | 1.662976 | 1.652x |
| 26 | p1024d1024 | gate_up | 33408 | 512 | 6144 | 75 | 1.076224 | 2.148864 | 0.501x |
| 27 | p1024d1024 | down | 33408 | 6144 | 256 | 75 | 3.541248 | 2.275584 | 1.556x |
