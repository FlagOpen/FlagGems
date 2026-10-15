# SM90 block-FP8 matrix multiplication

The public API remains `flag_gems.ops.w8a8_block_fp8_matmul.w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype=BF16, *, tuned=True)`.
No SGLang import, request metadata or serving layout is involved.

A is a contiguous `[M,K]` E4M3FN matrix; B is contiguous `[N,K]`.
For the exact SM90 K32 path, As is numeric FP32 `[M,K/32]` and Bs is
numeric FP32 `[ceil(N/32),K/32]`; strided scales are supported. Output is
BF16 or FP32. Encoded UE8M0 bytes must be decoded before calling this API.
The specialized path requires `(block_n,block_k)=(32,32)` and SM90.
Other shapes, dtypes, group sizes and devices keep the existing generic/
legacy-YAML selection. Existing six-field nearest-M YAML entries and
exact-M Hopper entries retain their behavior.

There are 162 explicit H100 `(M,N,K)` overrides in the private
`_w8a8_block_fp8_configs.py` data module. They select row/column tiles,
SwapAB, warps, stages, Split-K and static shape metadata. Selection does
not interpolate between these overrides. Each entry retains its reviewed
configured-native baseline; `tuned=False` selects that baseline for the
same finite shapes. This is a reproducible control, not a second matrix
interface. Legacy YAML configurations for other shapes remain separate
and unchanged.

Each K32 group applies its own FP32 scales. Split-K writes FP32 partials
and reduces them with the original reduction order. Numeric kernel bodies
are carried from the tested operator port without algebraic changes.
They derive from SGLang's Apache-licensed implementation and the subsequent
H100 optimization work; attribution is retained in source. Repository
ownership follows the general matrix/scale interface.

Run from a CUDA checkout with `PYTHONPATH=src`:

```bash
python -m pytest -q tests/test_w8a8_block_fp8_config.py tests/test_dsv41_block_fp8_matmul.py
python benchmark/sm90_block_fp8_graph.py --out fp8-graph.json --rounds 2 --rep-ms 100
```

Tests use an independent FP32 block-scale reference and byte-exact native
regression, including changed-input CUDA Graph replay and strided scales.
Byte equality is checked with model-style power-of-two scales represented
as numeric FP32; it is not promised for arbitrary scale distributions.
The independent reference uses rtol=.01/atol=.01. Benchmark timing includes
the complete public wrapper and Split-K reduction, excludes JIT/input
creation, and retains every shape. Two-round mean latencies form speedup
ratios; the summary uses an equal-shape geometric mean.

The pre-split H100 measurement covered 162 shapes: 1.410x geometric mean,
range 1.042–3.795x. Its immutable numerical source is
[the measured pre-split port](https://github.com/CherryLemon/FlagGems-sglang/commit/b188ef6b9de41138331299651aa23f3e11fb8187).
The engineering split requires its own numerical/Graph parity verification;
those historical microbenchmarks are not a newly measured serving gain.
The SGLang adapter is maintained separately in
[FlagGems-sglang #161](https://github.com/flagos-ai/FlagGems-sglang/pull/161).
