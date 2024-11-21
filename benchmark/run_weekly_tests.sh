#!/bin/bash
# set -x

ts=$(date +"%Y-%m-%d_%H-%M-%S")
dirp=logs_perf_${ts}${1}
mkdir -p $dirp
echo "saving logs to: "$dirp

echo "starting date: "$ts

pytest --level core --record log -sv 2>&1 | tee  ${dirp}/perf_test_log_${ts}.log
cp result--level_core--record_log-sv.log   ${dirp}/perf_test_result_${ts}.json
python summary_for_plot.py  result--level_core--record_log-sv.log

echo "finish date: "$(date +"%Y-%m-%d_%H-%M-%S")
echo "saved logs to: "$dirp
