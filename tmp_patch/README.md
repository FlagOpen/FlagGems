# 临时pacth说明

- 修改了 benchmark/performance_utils.py，运行时需将本目录下的 performance_utils.py 文件覆盖至 benchmark/ 下
- 覆盖后，运行程序前，需要关闭自动清理编译缓存功能（保留kcore.bin内容）：`export TRITON_TX8BE_CLEANUP=0`
- 修改内容及逻辑：对一个case，测试完成后，将编译产物由 /root 下移动到 /data/baai-benchmark-test-case 目录下，因为内容比较多比较大，挂载点 /root 一共只有几百GB，而 /data 挂载点有42T空间
- benchmark运行完成后，执行 `python get_board_run_time.py` 会遍历 /data/baai-benmark-test-case 下的所有case，使用 eng_test 程序加载kcore.bin，上板运行，统计出计算时间
- run_engtest.sh 为执行单个 kcore.bin 的shell脚本，由 get_board_run_time.py 中调用
