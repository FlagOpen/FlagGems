#!/bin/bash
# set -x

ts=$(date +"%Y-%m-%d_%H-%M-%S")
dirp=logs_perf_${ts}${1}
mkdir -p $dirp
echo "saving logs to: "$dirp

pip install pytest-xdist pytest-timeout
echo "starting date: "$ts

pytest --level core --record log -sv --durations=0 --timeout=3600  2>&1 | tee  ${dirp}/perf_test_log_${ts}.log

RESF="result--level_core--record_log-sv--durations_0--timeout_3600.log"
cp $RESF   ${dirp}/perf_test_result_${ts}.json
python summary_for_plot.py  ${dirp}/perf_test_result_${ts}.json

echo "finish date: "$(date +"%Y-%m-%d_%H-%M-%S")
echo "saved logs to: "$dirp
