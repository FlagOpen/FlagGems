---
title: QA Acceptance Testing
weight: 60
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->


# QA Acceptance Testing

`.github/workflows/ops-test.yaml` is *FlagGems*' full-range operator QA
acceptance workflow. It runs accuracy and performance tests on the enabled
hardware backends selected for a run and, by default, uploads result archives
to the op-monitor dashboard. This page describes the test environment, the
configuration files that drive the workflow, and how to reproduce a run locally.

## 1. Test environment

Each backend runs on a self-hosted runner with real accelerator hardware.
Backends with `use_container: true` use a dedicated job container with a
backend-specific image, mounted volumes, and runtime options. `PPU-ZW810E`
has `use_container: false` and runs natively on its runner, using the driver
stack and SDK installed there.

The backends currently enabled in `.github/configs/backends/config.yaml`
are:

| Config file (`name`) | Vendor (`vendor`) | Container image (if used) |
| --- | --- | --- |
| `H800` | Nvidia | `flagos-runtime-nvidia-cuda13.3` |
| `910B` | Ascend | `flagos-runtime-ascend-cann9.0.0` |
| `C550` | Metax | `flagos-runtime-metax-maca3.8.1.3` |
| `BW1000` | Hygon | `flagos-runtime-hygon-dtk26.04` |
| `MTT-S5000` | MThreads | `flagos-runtime-mthreads-musa5.2.0` |
| `BI-V150` | Iluvatar | `flagos-runtime-iluvatar-corex4.5.0` |
| `PPU-ZW810E` | THead | Native runner (no container) |
| `MLU590-M9DE` | Cambricon | `flaggems-test-cambricon-mlu590-m9de` |
| `P800` | KunLunXin | `flagos-runtime-kunlunxin-xre5.37.1` |
| `W7900D` | AMD | `flaggems-test-amd-w7900d` |

Backend configs include:
- `runner_labels`: the self-hosted runner label the job is pinned to.
- `use_container`: whether the job uses a container or runs natively.
- For container backends, `container_volumes` / `container_options` specify
  host mounts (`/data`, `/home`, `/tmp`) and flags such as `--privileged`,
  `--ipc=host`, and vendor-specific device access (e.g. `--gpus all` on Nvidia).
- `test_gpus`: records device IDs in the config. The current workflow uses
  the availability probe's `AVAILABLE_GPUS` output for `run_tests.py` instead.
- `gpu_check_script`: a per-vendor script (e.g. `tools/gpu_check_nvidia.sh`)
  that waits for a device with enough free memory before the run starts.
- `output_base_dir`: where result archives are written on the runner host.

In the job environment (container or native), the workflow installs *FlagGems*
with the `test` extra (`pip install .[test]`), which brings in `pytest`, `coverage`,
`pandas`, `openpyxl`, and other reporting dependencies declared in
`pyproject.toml`.

## 2. Test configuration

### Backend selection

`.github/configs/backends/config.yaml` is the master switch: it lists every
backend and whether it is `enabled`. The workflow's `prepare` job reads this
file, loads the matching `.github/configs/backends/<NAME>.yaml` for each
enabled backend, and builds the job matrix from it. A `workflow_dispatch`
input (`vendors`) can further narrow the matrix to a comma-separated subset
of vendor names (case-insensitive), e.g. `Ascend,Nvidia`.

### Operator selection

Operators come from `conf/operators.yaml`. The latest entry in each
operator's `stages` history determines its current stage (`alpha`, `beta`,
`stable`, or `removed`). `tools/run_tests.py` filters by `--stages` (default
`stable`); the workflow maps its `ops` input as follows:
- empty (the default) -> `--stages all`, covering `alpha`, `beta`, and
  `stable`, excluding `removed`.
- a comma-separated list of operator IDs (e.g. `abs,sum,add`) -> `--ops
  <list>`, testing only those operators.

Scheduled runs are created by an external cron job via `workflow_dispatch`;
`ops-test.yaml` itself does not declare a `schedule` event.

### Result publishing

Two `workflow_dispatch` inputs gate the side effects of a run:
- `upload_log`: leave empty to publish the result archive to op-monitor via
  `secrets.OPMON_URL` / `secrets.OPMON_UPLOAD_KEY`; set any value to skip
  the "Publish results to op-monitor" step.
- `send_feishu`: leave empty to send a Feishu notification via
  `.github/scripts/notify_feishu.py`; set any value to skip it.

## 3. Reproducing a run locally

The workflow is just a thin wrapper around `tools/run_tests.py` plus a few
report-generation scripts, so the same steps can be run directly on any
machine with the target accelerator, driver stack, and matching vendor
PyTorch/Triton environment already installed.

1. Install *FlagGems* with the test extras:

```shell
pip install .[test]
```

2. Make sure the accelerator is idle and has enough free memory (the
   vendor's `gpu_check_script`, e.g. `tools/gpu_check_nvidia.sh`, waits for
   this automatically in CI). Identify which device IDs you want to test,
   e.g. `0,1,2,3`.

3. Clear any stale Triton/FlagGems cache, then run the test scheduler. This
   mirrors the "Run ops tests" step in the workflow:

```shell
rm -rf ~/.triton/cache && rm -rf ~/.flaggems

OUTPUT_DIR="logs_results_$(date +%Y%m%d_%H%M)"
# Run all active-stage operators on 4 devices, matching an empty workflow ops input.
python3 tools/run_tests.py \
    --gpus 0,1,2,3 \
    --stages all \
    --dump-output \
    --output-dir "${OUTPUT_DIR}"
```

   To reproduce the `ops` input instead of a full-stage run, pass
   `--ops` with a comma-separated operator list, matching the workflow's
   non-empty `ops` input:

```shell
python3 tools/run_tests.py --gpus 0,1,2,3 --ops "abs,sum,add" \
    --dump-output --output-dir "${OUTPUT_DIR}"
```

4. Generate the same summary artifacts the CI job produces (operator labels,
   CSV, and HTML reports), reusing `OUTPUT_DIR` from step 3:

```shell
python3 tools/add_labels.py "${OUTPUT_DIR}"
python3 tools/psum_text -o "${OUTPUT_DIR}/result.csv"  "${OUTPUT_DIR}"
python3 tools/psum_html -o "${OUTPUT_DIR}/result.html" "${OUTPUT_DIR}"
```

5. Inspect `${OUTPUT_DIR}/summary.json` for aggregated pass/fail results, or
   open `${OUTPUT_DIR}/result.html` for the human-readable report. Each
   operator's JSON outputs, when produced, are stored in
   `<op_name>/accuracy_result.json` and `<op_name>/performance_result.json`.
   Missing or timed-out tests may not produce the corresponding JSON file.
   Passing `--dump-output` also saves stdout/stderr logs.

To trigger the same run in CI instead of reproducing it locally, dispatch
the workflow manually (**Actions -> ops-test -> Run workflow**) and fill in
the inputs described above: `branch`, `vendors`, `ops`, `upload_log`, and
`send_feishu`. `branch` selects the checkout ref and defaults to `master`;
set it to `kernelgen-dev` explicitly when testing that branch.

See [Testing Python Operators](../unittest) for the lower-level `pytest`
options that `tools/run_tests.py` drives under the hood.
