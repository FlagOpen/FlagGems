# 已验证的 benchmark skip

`benchmark/verified_case_skips.json` 保存 2026-10-22 的 95 项 reference-only 复测中确认的 370 个 case：昇腾 350 个 OOM、4 个 put BF16 不支持，海光 8 个 OOM，沐曦 8 个 OOM。原始 Gems 为 47246ae229e1c795f63dc6624a407ace2ef54747，KGS 为 95a5f62b23619c35d1523b53afb0c027cff40a63；原始归档为 full-reference-review-20261022-RCw8L3/results。

每行依次为 vendor、case_id、metadata SHA-256、原因。metadata 是 dtype、shape、params 三字段的 JSON，采用 sort_keys=True、separators=(",", ":") 后计算 SHA-256。只有所有字段匹配才在输入分配前触发 pytest.skip；shape、参数或 dtype 改变会使规则失效。list-cases 仍保留原 case，不删除 workload。规则仅覆盖 benchmark，不推导到 correctness。规则不是通用硬件容量声明，换显存容量或运行环境后应重新验证并删除不再适用的条目。

不捕获运行时 OOM 后伪装成功，不跳过 API/kernel 缺失、形状错误、输出 dtype 不一致、共享内存超限或未知错误。全 SKIP 不等于 reference/preflight 通过。reference、preflight 按 case 记录 SKIP 并继续后续 case。

同时迁移 master@4897cc306 的相关已有策略：EXTRA_INT_DTYPES 服从 support_int64，tsingmicro 的 linalg_svdvals FP32 skip，以及 enflame 的 linalg_ldl_factor FP64 skip。保留 kernelgen-dev 的注入接口及已有测试覆盖。master 通用 foreach bool CPU reference skip 对应的新测试结构在此分支尚不存在，当前 per-operator foreach abs 没有 bool 正常测例，因此不引入整套新测试或无效规则。master 缩小 foreach shape 属于 workload 变更，不作为 skip 同步。
