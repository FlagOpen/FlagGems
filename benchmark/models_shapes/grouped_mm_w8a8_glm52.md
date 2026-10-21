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
every call, optionally compacts active expert IDs in the same metadata kernel,
and writes C directly. There is no temporary GEMM output or copy.
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
The shared accuracy suite passed 32 tests, including changing sparse expert
routes, multi-tile experts and masked N/K tails. This full-replay accuracy
statement applies to the deterministic operands/scales described above.

Sparse INT8 batches use one pipeline stage: long-K GEMMs use 16x64x256 tiles,
and short-K, wide-N GEMMs use 16x128x128 tiles. Selected sparse batches build
active expert IDs alongside the expert prefix and bound the launch grid by
min(M,E). Both metadata operations are measured; there is no host count readback
or extra metadata launch. The stage count is an explicit kernel specialization
key. Other workloads retain their prior tile/stage selection.

| Projection | Records | Call weight | mctlass weighted ms | FlagGems weighted ms | Weighted total-time ratio | Faster cases |
|---|---:|---:|---:|---:|---:|---:|
| all | 28 | 3240600 | 0.181226 | 0.116800 | 1.552x | 15/28 |
| gate_up | 14 | 1620300 | 0.130644 | 0.134553 | 0.971x | 1/14 |
| down | 14 | 1620300 | 0.231808 | 0.099047 | 2.340x | 14/14 |

The weighted total-time ratio is **1.552x**; the distinct
call-weighted arithmetic mean of per-case speedups is **1.692x**.
Both aggregate metrics exceed 1.3x. This is not a per-case guarantee:
gate/up's weighted total-time ratio remains 0.971x, and large-M gate/up
is still below the native baseline. This is an operator comparison, not model
inference speedup.

Compared with the same captured replay at commit `8566132b`, weighted candidate
latency falls from 0.145347 to 0.116800 ms
(19.64% reduction). The native baseline's weighted latency changes
by less than 0.1%, supporting comparison across the two three-run measurements.

| Projection | M | Previous ms | Current ms | Latency reduction | Current mctlass / FlagGems |
|---|---:|---:|---:|---:|---:|
| gate_up | 45 | 0.107520 | 0.098048 | 8.81% | 1.070x |
| gate_up | 117 | 0.183808 | 0.175360 | 4.60% | 0.978x |
| down | 45 | 0.122112 | 0.071424 | 41.51% | 2.014x |
| down | 117 | 0.170240 | 0.125440 | 26.32% | 2.765x |

Maximum three-run latency span `(max/min - 1)` is
0.39% for mctlass and
2.98% for FlagGems.
Raw JSON, CSV, accuracy records, before/after comparisons and source hashes are
retained on the server in
`/data/ldc/fzy/group-mm-mixed-20261009/smallm-opt-20261021`.

All rows have E=257. Speedup = mctlass / FlagGems; less than 1 means slower.

| # | Scenario | Projection | M | N | K | Count | mctlass ms | FlagGems ms | Speedup |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | p4096d1024 | gate_up | 45 | 512 | 6144 | 921600 | 0.104960 | 0.098048 | 1.070x |
| 1 | p4096d1024 | down | 45 | 6144 | 256 | 921600 | 0.143872 | 0.071424 | 2.014x |
| 2 | p1024d1024 | gate_up | 117 | 512 | 6144 | 441600 | 0.171520 | 0.175360 | 0.978x |
| 3 | p1024d1024 | down | 117 | 6144 | 256 | 441600 | 0.346880 | 0.125440 | 2.765x |
| 4 | p4096d1024 | gate_up | 36 | 512 | 6144 | 76800 | 0.072448 | 0.076544 | 0.946x |
| 5 | p4096d1024 | down | 36 | 6144 | 256 | 76800 | 0.135424 | 0.044032 | 3.076x |
| 6 | p1024d1024 | gate_up | 126 | 512 | 6144 | 62400 | 0.130048 | 0.154880 | 0.840x |
| 7 | p1024d1024 | down | 126 | 6144 | 256 | 62400 | 0.397568 | 0.101376 | 3.922x |
| 8 | p1024d1024 | gate_up | 72 | 512 | 6144 | 62250 | 0.148992 | 0.162304 | 0.918x |
| 9 | p1024d1024 | down | 72 | 6144 | 256 | 62250 | 0.213504 | 0.098304 | 2.172x |
| 10 | p1024d1024 | gate_up | 108 | 512 | 6144 | 43200 | 0.168192 | 0.173056 | 0.972x |
| 11 | p1024d1024 | down | 108 | 6144 | 256 | 43200 | 0.324608 | 0.122112 | 2.658x |
| 12 | p1024d1024 | gate_up | 36864 | 512 | 6144 | 1875 | 1.235200 | 2.331264 | 0.530x |
| 13 | p4096d1024 | gate_up | 36864 | 512 | 6144 | 4800 | 1.105664 | 2.204928 | 0.501x |
| 14 | p1024d1024 | down | 36864 | 6144 | 256 | 1875 | 3.678976 | 2.540928 | 1.448x |
| 15 | p4096d1024 | down | 36864 | 6144 | 256 | 4800 | 3.672576 | 2.496768 | 1.471x |
| 16 | p1024d1024 | gate_up | 90 | 512 | 6144 | 4650 | 0.161024 | 0.171520 | 0.939x |
| 17 | p1024d1024 | down | 90 | 6144 | 256 | 4650 | 0.257536 | 0.114688 | 2.246x |
| 18 | p1024d1024 | gate_up | 9216 | 512 | 6144 | 600 | 0.616448 | 1.537280 | 0.401x |
| 19 | p1024d1024 | down | 9216 | 6144 | 256 | 600 | 1.485440 | 0.718336 | 2.068x |
| 20 | p1024d1024 | gate_up | 81 | 512 | 6144 | 300 | 0.152320 | 0.164864 | 0.924x |
| 21 | p1024d1024 | down | 81 | 6144 | 256 | 300 | 0.245248 | 0.102912 | 2.383x |
| 22 | p1024d1024 | gate_up | 16704 | 512 | 6144 | 75 | 0.833536 | 1.421312 | 0.586x |
| 23 | p1024d1024 | down | 16704 | 6144 | 256 | 75 | 1.873920 | 1.674496 | 1.119x |
| 24 | p1024d1024 | gate_up | 17856 | 512 | 6144 | 75 | 0.774656 | 1.365504 | 0.567x |
| 25 | p1024d1024 | down | 17856 | 6144 | 256 | 75 | 2.747904 | 1.660160 | 1.655x |
| 26 | p1024d1024 | gate_up | 33408 | 512 | 6144 | 75 | 1.075968 | 2.141952 | 0.502x |
| 27 | p1024d1024 | down | 33408 | 6144 | 256 | 75 | 3.541120 | 2.276352 | 1.556x |
