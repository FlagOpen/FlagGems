time pytest -s benchmark/performance_flagperf.py -k "test_perf_groupnorm_backward"

echo "------------------------------------------------------------------------------"
time pytest -s benchmark/test_reduction_perf.py -k "test_perf_groupnorm_backward"
