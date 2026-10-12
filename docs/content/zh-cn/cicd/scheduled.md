---
title: 定时与按需测试
weight: 20
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

<!--
# Scheduled and On-Demand Testing

Beyond per-PR checks, FlagGems runs broader test sweeps on a schedule to
catch regressions that only show up across the full operator suite or across
vendors that aren't touched by a given PR.
-->
# 定时与按需测试

除了针对每个 PR 的检查外，FlagGems 还会按计划运行更大范围的测试，
以捕获那些只在全量算子测试或某个 PR 未涉及的厂商后端上才会暴露的回归问题。

<!--
## `daily.yaml` — full-suite regression on NVIDIA

**Trigger:** cron `3 16 * * *` (00:03 Beijing time, daily); also manual
`workflow_dispatch`.

Runs the entire operator test suite (not just changed operators) on the
NVIDIA runner:

- `cpp-op` — the full C++ extension test suite.
- `python-op` — `tools/test-op.sh` with `CHANGED_FILES=__ALL__`, which
  expands to every `tests/test*.py` file. Also collects coverage and
  uploads it as the `op-ut-coverage` artifact.
- `examples` — `tools/test-examples.sh` against all example/model tests.
-->
## `daily.yaml` — NVIDIA 平台全量回归测试

**触发条件：** cron 表达式 `3 16 * * *`（每天北京时间 00:03）；此外支持手动
触发（`workflow_dispatch`）。

在 NVIDIA runner 上运行完整的算子测试套件（而非仅测试变更的算子）：

- `cpp-op` —— 完整的 C++ 扩展测试套件。
- `python-op` —— 以 `CHANGED_FILES=__ALL__` 运行 `tools/test-op.sh`，
  该设置会展开为 `tests/test*.py` 下的所有测试文件。同时收集覆盖率数据
  并作为 `op-ut-coverage` 构件（artifact）上传。
- `examples` —— 对所有示例/模型测试运行 `tools/test-examples.sh`。

<!--
## `coverage.yaml` — publish coverage to the docs site

**Trigger:** `workflow_run` after `daily` completes; also manual
`workflow_dispatch` with a `run_id` input.

Downloads the `op-ut-coverage` artifact produced by `daily.yaml`, unpacks
the HTML coverage report and summary markdown, commits them under
`docs/static/coverage/<date>` and `docs/content/en/references/test/unit/`,
and pushes directly to the `gh-pages` branch. `hugo-site.yaml` then rebuilds
and republishes the site in response to that push.
-->
## `coverage.yaml` — 将覆盖率发布到文档站点

**触发条件：** `daily` 工作流执行完成后触发（`workflow_run`）；此外支持手动
触发（`workflow_dispatch`），需提供 `run_id` 参数。

下载 `daily.yaml` 生成的 `op-ut-coverage` 构件，解压 HTML 覆盖率报告和
汇总 markdown 文件，将其提交到 `docs/static/coverage/<date>` 和
`docs/content/en/references/test/unit/` 目录下，并直接推送到 `gh-pages`
分支。`hugo-site.yaml` 随后会响应该推送，重新构建并发布站点。

<!--
## `ops-test.yaml` — multi-vendor QA acceptance testing

**Trigger:** `workflow_dispatch` only, with optional `branch`, `vendors`,
`ops`, `upload_log`, and `send_feishu` inputs. GitHub Actions' native
`schedule` trigger isn't used — it was observed to delay or drop runs by
1-5 hours under load, which isn't acceptable for GPU test scheduling.
Instead, `.github/scripts/trigger_workflow.sh` is installed as a crontab job
on an always-on control-plane server; by default it fires `30 21 * * 3,6`
(Beijing time, Wednesday and Saturday) and calls the GitHub API to dispatch
this workflow, so GitHub still allocates the runners and executes it. The
script retries on failure and checks for a duplicate run before retrying, to
avoid double-triggering.

Runs the full (or a filtered) operator suite across every enabled vendor
backend for QA acceptance testing — of a scheduled sweep via the external
dispatcher, or of a specific branch/vendor/operator set via a manual run
before a release. The `prepare` job resolves the checkout ref (defaulting
to `master`) and builds the backend matrix from
`.github/configs/backends/config.yaml` plus one YAML file per backend,
filtered by the `vendors` input (case-insensitive, empty = all enabled
vendors); the `ops` input likewise narrows the run to specific operators
(empty = every stage via `--stages all`).

Backends are split into two execution modes:

- **Container-based** (`test-container`) — runs inside the backend's Docker
  image, for vendors whose SDK is distributed as a container. Job timeout
  is 1440 minutes (24h) to accommodate the longest vendor runs (e.g. Ascend
  910B).
- **Native** (`test-native`) — runs directly on the self-hosted runner
  without a container, with an 1080-minute (18h) job timeout.

Each matrix job attempts checkout up to three times (the initial attempt plus
at most two retries, waiting 30s before each retry),
installs FlagGems, checks GPU availability, runs `tools/run_tests.py`
scoped by `--ops` or `--stages all`, summarizes results with
`add_labels`/`psum_text`/`psum_html`, packages them into a per-vendor
archive, and — unless disabled via the `upload_log`/`send_feishu` inputs —
uploads the archive to the internal op-monitor service and posts a Feishu
notification with the outcome.
-->
## `ops-test.yaml` —— 多厂商 QA 验收测试

**触发条件：** 仅支持手动触发（`workflow_dispatch`），可选参数包括
`branch`、`vendors`、`ops`、`upload_log`、`send_feishu`。该工作流不使用
GitHub Actions 原生的 `schedule` 触发器——因为在负载较高时，原生定时触发
曾出现延迟甚至丢失运行的情况（延迟 1-5 小时），这对于 GPU 测试的调度而言
是不可接受的。取而代之的方案是：在一台常驻运行的控制面服务器上部署
`.github/scripts/trigger_workflow.sh` 作为 crontab 任务，默认按
`30 21 * * 3,6`（每周三、周六北京时间 21:30）的节奏调用 GitHub API 来
触发本工作流，实际的 runner 分配和执行仍由 GitHub 完成。该脚本在失败时
会自动重试，并在重试前检查是否已有重复运行，以避免重复触发。

针对所有已启用的厂商后端运行完整（或经过筛选）的算子测试套件，用于 QA
验收测试——既可以是通过外部调度脚本发起的定期全量测试，也可以是发布前
针对指定分支/厂商/算子手动发起的测试。`prepare` 作业会解析代码检出引用
（默认为 `master`），并基于 `.github/configs/backends/config.yaml` 以及
每个后端对应的独立 YAML 文件构建后端矩阵，根据 `vendors` 输入参数进行
筛选（不区分大小写，留空表示所有已启用的厂商）；`ops` 输入参数同样用于
将测试范围缩小到指定算子（留空则通过 `--stages all` 运行全部测试阶段）。

各后端按两种执行方式划分：

- **基于容器**（`test-container`） —— 在后端对应的 Docker 镜像内运行，
  适用于以容器形式分发 SDK 的厂商。作业超时时间为 1440 分钟（24 小时），
  以适配耗时最长的厂商测试（例如 Ascend 910B）。
- **原生运行**（`test-native`） —— 直接在自托管 runner 上运行，不使用
  容器，作业超时时间为 1080 分钟（18 小时）。

矩阵中的每个作业最多尝试检出代码 3 次（首次失败后最多重试 2 次，
每次重试前等待 30 秒），随后安装
FlagGems、检查 GPU 可用性、运行 `tools/run_tests.py`（通过 `--ops` 或
`--stages all` 参数控制测试范围）、使用
`add_labels`/`psum_text`/`psum_html` 汇总结果，并将结果打包为每个厂商
对应的压缩包；除非通过 `upload_log`/`send_feishu` 输入参数禁用，还会将
压缩包上传到内部 op-monitor 服务，并发送包含结果的飞书通知。

<!--
## `command.yaml` — on-demand `/test` command

**Trigger:** `issue_comment` created on a PR, matching `/test
<operator>:<runner>`.

Lets any PR participant with the right permissions request an ad-hoc test
of a single operator on a specific runner (e.g. `/test
constant_pad_nd:mthreads`) without waiting for the operator to have a
`vendor/*` label or for a full CI run. For operators that already have
history, it runs the test before and after the PR's changes and posts a
comparison; for brand-new operators, it posts a single-run report. Results
(and full logs) are attached as a PR comment and an uploaded artifact.
-->
## `command.yaml` — 按需 `/test` 命令

**触发条件：** PR 下创建的 `issue_comment` 事件，且评论内容匹配
`/test <算子名>:<runner>` 格式。

允许拥有相应权限的任何 PR 参与者，针对单个算子在指定 runner 上发起临时测试
（例如 `/test constant_pad_nd:mthreads`），而无需等待该算子被打上 `vendor/*`
标签或等待完整的 CI 运行。对于已有测试历史的算子，会在 PR 改动前后分别运行
测试并发布对比结果；对于全新算子，则只发布单次运行的报告。测试结果（及完整
日志）会以 PR 评论和上传构件的形式提供。

<!--
## Manually dispatched utility workflows

These workflows only have a `workflow_dispatch` trigger (`random-test.yaml`,
`probe-node.yaml`, `review-stats.yaml`); see the English documentation.
-->
## 手动触发的辅助工作流

以下工作流只有 `workflow_dispatch` 触发器，需要在 **Actions** 页面点击
**Run workflow**，或通过 `gh workflow run` 手动运行。

### `random-test.yaml` —— 在指定 runner 上测试指定算子

**触发方式：** `workflow_dispatch`，包含三个必填输入：

| 输入 | 含义 | 示例 |
|---|---|---|
| `runner` | 运行所用的 runner 标签 | `a100`、`h100`、`h20`、`ascend`、`cann9`、`metax`、`maca381`、`enflame`、`cambricon` |
| `operators` | 以逗号分隔的算子名 | `add,mul,softmax` |
| `gpus` | 以逗号分隔的 GPU 编号 | `0,1` |

```bash
gh workflow run random-test.yaml \
  -f runner=a100 -f operators=add,mul,softmax -f gpus=0,1
```

`check-permission` 作业会先检查触发者是否属于 `FlagGems` 团队，不属于则跳过
测试作业。随后 `test-operator` 作业把 runner 标签映射为厂商和 backend
profile（例如 `a100`/`h100`/`h20` → `nvidia-cuda133`，`ascend`/`cann9` →
`ascend-cann900`；标签未知时直接用标签本身作为两者的值），检出代码，运行对应
厂商的 GPU 检查脚本和 `setup-flaggems`，再执行
`tools/run_tests.py --ops <operators> --gpus <gpus>`。`results/` 目录会作为
名为 `random-test` 的构件上传。

`operators` 和 `gpus` 会直接拼接到命令行，因此其中不要带空格。如需在本地复现，
可直接运行 `tools/run_tests.py`。

### `probe-node.yaml` —— 探测自托管 runner 的环境

**触发方式：** `workflow_dispatch`，有一个必填输入 `runner`（runner 标签，
默认 `h100`）。

收集自托管 runner 的环境信息，用于为其选择合适的 backend profile（例如
`nvidia-cuda133` 还是 `nvidia-cuda128`）。它只要求 runner 上有 bash，也不会
检出仓库。输出内容包括：

- 系统与 Python：`uname -a`、`python3 --version`
- 容器 / 虚拟化：`/.dockerenv`、`/proc/1/cgroup`、`systemd-detect-virt`、
  PID 1
- 内存：`free -h` 以及 cgroup 内存上限
- GPU：`nvidia-smi`，以及每张卡的显存总量 / 已用 / 空闲
- CUDA 工具链：`nvcc --version` 和 `/usr/local/cuda*`

输出会写入作业日志和该次运行的 **Summary** 页面（`$GITHUB_STEP_SUMMARY`）。
该探测是只读的，且没有团队成员权限检查。GPU 与 CUDA 相关探测仅针对 NVIDIA；
在其他厂商的机器上这些项会显示 "not found"，其余部分仍然有效。

### `review-stats.yaml` —— 拉取请求分析报告

**触发方式：** `workflow_dispatch`，有两个可选输入 `report_date_start` 和
`report_date_end`，格式均为 `dd/MM/YYYY`。

在 `ubuntu-latest` 上运行，使用
[`AlexSim93/pull-request-analytics-action`](https://github.com/AlexSim93/pull-request-analytics-action)
统计 `flagos-ai/FlagGems` 在给定日期范围内的拉取请求数据（评审响应时间、活跃度
等指标）。工作时间配置为 `Asia/Shanghai` 时区的 9:00–23:00。报告会以 issue
的形式创建在 `flagos-ai/FlagGems` 仓库中，因此该作业需要 `issues: write`
权限（以及对 contents、pull requests、statuses 的读权限）。

```bash
gh workflow run review-stats.yaml \
  -f report_date_start=01/09/2026 -f report_date_end=30/09/2026
```

<!--
## Feishu and monitoring integrations

- `ci-report-feishu.yaml` reports every `rule-check` completion to a Feishu
  Bitable for tracking pass/fail trends over time.
- `ops-test.yaml` and `command.yaml`'s failure paths send Feishu chat
  notifications via `.github/scripts/notify_feishu.py`.
- `ops-test.yaml` also uploads results to an internal "op-monitor" HTTP
  service for longer-term dashboards.

These are observability workflows; they do not gate merges.
-->
## 飞书与监控集成

- `ci-report-feishu.yaml` 会将每一次 `rule-check` 的执行结果上报到飞书
  多维表格（Bitable），用于跟踪长期的通过/失败趋势。
- `ops-test.yaml` 和 `command.yaml` 的失败处理路径会通过
  `.github/scripts/notify_feishu.py` 发送飞书群消息通知。
- `ops-test.yaml` 还会将结果上传到内部的 "op-monitor" HTTP 服务，用于
  生成长期监控看板。

这些均属于可观测性（observability）相关的工作流，并不会阻塞 PR 的合并。
