---
title: Benchmark reference-only
weight: 50
---

# Benchmark reference-only：检查原始 core baseline

`--reference-only` 仅用于 benchmark，默认 level 为 `core`，无需显式传入；可通过 `--level` 覆盖。普通 benchmark 仍默认 `comprehensive`。该模式通过 `get_latency(self.torch_op, *args, **kwargs)` 复用 forward/backward 和 fresh-input 准备，在共享计时层使用 `measure_calls(fn, warmup_calls=0, repeat_calls=1)`：不预热，只执行一次。不需要 candidate，不使用 override，不做正确性比较、Profile 或加速比计算；不能据此声称正确性 pytest 的 reference 或完整测试链路已经通过。

```bash
pytest -q benchmark/test_negative.py --reference-only --output benchmark-reference.json
```

可沿用单独一次 `--list-cases --level core` 得到的 `--case-id` 精确重放；列举用例时需与实际执行 level 匹配。KGS review 会使用整个 core 集合，不缩减 dtype 或 workload。输出每次覆盖写入，不累计旧结果。先核对 `flag_gems.__file__` 来自预期 checkout，避免旧 editable install 导致使用另一版本。

## 复用与边界

标准 case-based Benchmark 复用 `build_inputs()`、`unpack_to_args_kwargs()` 和 `get_latency()`，明确传入原 `torch_op`。共享计时封装使用调用次数，不将 `--warmup/--iter` 的毫秒值当作次数；reference-only 不进入自动校准、重复采样或 CUDA Graph 捕获。它同步设备并记录单次 wall time，可能包含首次编译成本，只是就绪诊断，不是稳态 kernel latency。backward 沿用一次 forward 建图及一次 grad 调用；fresh-input 在测量前复制输入一次。结束后释放输入，不进入 candidate 的 `use_gems`、`gems_op` 或 override 分支。普通 benchmark 的计时模式、预热和重复测量完全不变。

没有 case builder 或自行覆盖 `run` 绕过共享执行入口的 benchmark，仍明确报告 `UNSUPPORTED`。原 `skip_native` 和 pytest skip 条件保留。该模式不是任意 Python 测试代码的沙箱。

无需装饰器或逐个 pytest 声明，也不比较方法是否与基类为同一个函数对象。`get_latency(op, ...)` 的统一契约就是测量传入的 `op`；自定义包装器应委托共享计时层，不得忽略参数而调用 Gems candidate 或自行绕过单次策略。这个封装不承诺控制任意第三方黑盒计时器。`OperatorBenchmark` 已复用此路径。reference-only 不调用同时测量两侧的 `_measure_input`；仅在该方法中定义的自定义 baseline 语义须迁入 `torch_op`/`get_latency` 的统一契约，不能假定本模式会执行它。

本接口不修改或执行正确性 pytest，不提供 correctness reference 的 marker、包装器或截断逻辑。KG 可在此后通过独立注入探针确认正确性 pytest 是否调用候选，生成候选后再执行原完整正确性比较。性能和正确性 reference 的 dtype、精度、设备、shape 可能不同，不能相互替代。

`--reference-only` 与 `--override`、`--override-config`、`--preflight-only`、`--profile-only`、`--list-cases`、`--query`、benchmark `--parallel` 及 xdist 并发互斥。不能与 `tests/` 下的正确性用例混跑。

## 报告

```json
{
  "schema_version": "flaggems.reference/v2",
  "phase": "timing",
  "status": "PASSED",
  "records": [{
    "nodeid": "benchmark/test_negative.py::test_negative",
    "operator": "negative",
    "case_id": "benchmark/test_negative.py::test_negative::core::float32::0",
    "latency_ms": 0.012,
    "status": "PASSED"
  }]
}
```

报告区分 `PASSED`、`FAILED`、`UNSUPPORTED`、`ALL_SKIP`、`NO_CASES`，成功测量保存诊断用 `latency_ms`，不含 speedup 或计时器内部调用 count。耗时须有限且非负；允许计时器返回分辨率内的零值，但零值不证明具有有效 headline performance。报告使用 `flaggems.reference/v2` 区别旧版不含测量值的报告，调用方需要同步更新校验。pytest 阶段级失败或跳过另带 `pytest_phase`。fixture/marker 或 case 循环外的 node-level skip 按原节点语义处理，之前的调用只保留为执行证据，不据此声称该 node 完整通过。全部跳过可能仍为 pytest exit code 0，因此调用方必须检查结构化状态，不能仅凭进程退出码判断就绪。

KGS 配套通过设备 slot 和隔离 worker 调用此命令，核对冻结 benchmark fingerprint 与 core case 覆盖，保存原报告；该执行事实不是模型审核结论，也不写入候选优化 ledger。KG 的 skip 选项不能绕过这项目标验证。

### 逐 case 失败信息

每个已选 case 保留 `case_id`、`ordinal`、`dtype`、`shape`、`params` 和执行状态。失败记录额外包含 `stage`（`build_inputs`、`benchmark_reference` 或 `synchronize`）与 `failure`（`category`、异常 `type`、原始 `message`、`traceback`），用于区分输入构造、原 baseline 测量和外层设备同步失败。共享计时封装内部的调用或同步异常都属于 `benchmark_reference`，由原 traceback 定位。`dtype` 来自 case 声明；backward 或原始 reference 内部可能转换 dtype，具体报错仍以原始异常为准。

`category` 是保守的诊断提示，不是设备能力表：`OUT_OF_MEMORY` 来自明确的 `torch.OutOfMemoryError`；`DTYPE_UNSUPPORTED` 仅识别明确的 PyTorch `not implemented for '<dtype>'` 信息；`API_MISSING` 用于访问缺失的 `torch` 模块或 ATen namespace/overload 属性；`NOT_IMPLEMENTED` 保留无法确定是整个 API、后端还是参数组合不支持的 `NotImplementedError`；参数、shape、索引及断言异常归为 `TEST_ERROR`，其他异常保留 `UNKNOWN`。明确的设备致命错误为 `DEVICE_ERROR`，宿主内存分配失败为 `HOST_OUT_OF_MEMORY`。不能从某一个 dtype 或某个输入失败推断整个算子不可用。

普通 case 错误在设备同步正常后继续下一个 case，包含 OOM、dtype 不支持、API 缺失、测试参数错误和未知普通异常。OOM 先释放异常栈持有的输入、回收对象并清理设备缓存，再同步检查。所有这些失败仍记录为 `FAILED`，遍历结束后 pytest 报告失败，不自动改成 SKIP。只有 OOM 和明确 dtype 不支持的 `failure.skip_eligible=true`，表示可据此制定针对平台/case 的显式 skip 规则；API 缺失、普通 dtype 不匹配、数值断言等均为 false。显式规则落地前，只有这两类错误也不能通过审核。

同步失败、明确设备致命错误、宿主内存耗尽或中断立即停止当前 node，剩余 case 标为 `NOT_RUN`。错误恢复若失败，另存 `recovery_failure` 并撤销该 case 的 skip 资格。case 执行内部触发的显式 pytest skip 只跳过当前 case，后续 case 继续；pytest fixture/marker 或 case 循环外的 node-level skip 仍按 pytest 节点语义处理。全部 case 跳过仍为 `ALL_SKIP`。收集、setup 或进程崩溃可能没有完整逐 case 报告，不能虚构结果。

## 验证范围

逐 case 收集改动基于 `kernelgen-dev@0c4c52bd`，不修改算子 correctness pytest。独立 host 转移测试 11 项通过；H20 现有运行时执行完整 reference-only 回归 68 项通过，覆盖固定调用次数、普通 benchmark 不变、fresh-input、backward、case 选择、skip/异常和配置互斥。

2026-10-22 真机验证代码为 `ba3f08bb`：摩尔原 foreach_erf core 测例收集到 28 PASSED、2 OUT_OF_MEMORY，失败后仍执行后续 case；原 bartlett_window 收集 10 个 TEST_ERROR，linalg_ldl_factor_ex 收集 4 个运行库错误，均保留失败。摩尔另用原生产 case 循环做设备 API 诊断，实际 BF16 不支持、缺失 Torch API、FP32 成功依次得到 FAILED/FAILED/PASSED，skip_eligible 分别为 true/false。昇腾原 kaiser_window 收集 15 个 backend 错误，均不可 skip。旧 _logcumsumexp 和 quantized_max_pool2d 的当前 pytest 缺少 case 接口，未进入 reference 循环，不计为通过。

配套 KG 经真实 KGS HTTP 接口完成正常 fmod_scalar 与失败 bartlett_window 的 prepare/review 验证；后者返回 REFERENCE_TEST_ERROR 及 10 项失败摘要。所有失败仍阻止下游优化；没有模型调用、没有安装或升级运行时依赖。较早 70 算子的结果属于旧实现，不能替代本轮验证。
