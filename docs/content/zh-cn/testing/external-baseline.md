---
title: 使用外部 NVIDIA 性能基线
weight: 35
---

# 使用外部 NVIDIA 性能基线

通过 `--base-data` 指定 ops-baseline 格式的 JSON，与当前 Gems 性能进行比较：

```bash
pytest benchmark/test_add.py --level core --record json \
  --base-data=/absolute/path/pytorch_baseline.json

python tools/run_tests.py --ops add --gpus 0 \
  --base-data /absolute/path/pytorch_baseline.json
```

pytest 建议使用 `--base-data=路径`，避免仓库外的路径影响早期 rootdir 解析。
批量运行器会先把路径解析为绝对路径。文件名不限，不自动下载数据。

记录按算子名、shape key、`torch.float32` 等 dtype 键组织，延迟使用
`latency_ms`；缺少该字段时也接受 `T_us` 并换算成毫秒。无效文件告警后关闭外部比较，
非正数、非有限数或非数字延迟不会使用。

包含 `config.inputs.args/kwargs` 时，匹配真实输入的嵌套结构、标量值与类型，以及
记录中的 tensor shape、dtype、stride 和 requires_grad，支持 ops-baseline PR #8
的 PyTorch 记录及其复合 shape key。显式参数与省略默认值不会自动合并。
具名 `config.inputs` 需要 reference callable 有可绑定的 Python 签名；无法绑定时回退。
没有输入元数据的旧记录仅按首个 tensor 的 shape key 和 dtype 匹配，不能检查其他
参数或布局，复杂算子应提供完整输入元数据。算子名精确匹配并保留下划线前缀，
不猜算子别名；多个记录同时匹配时放弃外部结果。

对于具名 `device` 参数，来源的 `cuda` / `cuda:0` 可匹配本次 benchmark 明确选择的
目标加速设备，例如 `npu`、`mlu:2`。CPU 位置、非默认来源 CUDA 索引、其他参数名和
非选定目标设备仍严格比较；不忽略 tensor 布局或辅助设备信息。
`torch.device` 实参统一描述为设备字符串。

命中后新增 `vs Base` 列，计算：

```text
原始比值 = 外部 NVIDIA 延迟 / 当前 Gems 延迟
compared_speedup = 原始比值 / 芯片折算系数
```

折算系数取 `_scaling_factors.chips`，根据当前 vendor/chip、基线瓶颈资源
`bottle_neck_unit`（mem/cuda/tensor）和 dtype 选择，优先 measured，其次 nominal。
缺少有效系数时保留原始比值。已知芯片名不匹配、匹配多个型号，或无法检测型号且该
厂商有多个条目时不使用系数。型号使用边界匹配，H20 不会命中 H200，允许 H20-3e
这样的后缀；厂商名显式兼容 ops-baseline 的 `t-head` 与 FlagGems 的 `thead`。
使用者应确保折算参考芯片与基线实际采集环境一致。

原有 benchmark JSON 的本机 `speedup`、Torch/Gems 延迟保留，新增外部
`compared_speedup`、数据路径、shape key、外部延迟、原始比值和折算系数。
`tools/run_tests.py` 最终汇总逐条优先使用 `compared_speedup`，未命中用原有比值；
保持现有算术平均口径，命中的明细额外保留 `native_speedup`。未命中记录原因。

该参数不会自动跳过本机原生实现。已有 `skip_native` 标记仍负责厂商基线缺失的场景，
这些场景可使用外部比值；外部也未命中时保持 N/A。正确性、preflight、reference-only、
profiling 和输入生成流程不受影响。完整格式与边界见英文同名文档。
