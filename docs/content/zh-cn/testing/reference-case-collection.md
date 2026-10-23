# Reference-only 逐 case 失败收集

`--reference-only` 遍历选中的 benchmark case；单个 case 的普通错误不再立即终止整个 pytest 节点。报告保留 case ID、dtype、shape、params、失败阶段、异常类型、消息和 traceback。全部遍历完成后，只要存在失败，节点及 reference 报告仍为失败，不将收集完成视为审核通过。

遇到明确的 `torch.OutOfMemoryError` 时，先保存文本证据，释放异常栈和输入引用，再执行 GC、设备缓存清理和同步；恢复成功才继续下一个 case。输入构造和 reference 调用中的 OOM 均适用。只有明确 OOM 和 dtype 不支持会标记 `skip_eligible=true`，该标记仅用于后续制定精确 skip 规则，不会自动跳过失败或修改测试。

同步失败、设备错误、主机 `MemoryError`、进程中断以及 OOM 恢复失败会停止遍历；后续 case 保持 `NOT_RUN`。进程崩溃或外部超时也不保证收集完整。API 缺失、参数错误和未知异常在同步成功后继续记录，但不能据此生成 skip 规则。

已有显式 pytest skip 继续保留为 `SKIP`，不影响后续 case；全部 skip 不算通过。本改动不改变正确性测试、候选注入、Preflight 或正常 benchmark 的执行语义。

无需设备的回归测试：`python3 -m pytest --noconftest -q tests/core/test_reference_failure_collection.py tests/core/test_verified_case_skips.py`。测试直接编译当前工作树的生产遍历方法并替换设备依赖，覆盖连续 OOM、输入构造 OOM、后续成功、恢复失败、同步错误、主机内存耗尽和显式 skip。
