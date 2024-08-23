#!/bin/bash

set -e

PR_ID=$1
PR_ID_DIR="PR_${PR_ID}_Coverage"

export PR_ID=168
export GITHUB_SHA=111
export GITHUB_RUN_ATTEMPT=222

ID_SHA_ATTEMPT="${PR_ID}-${GITHUB_SHA}-${GITHUB_RUN_ATTEMPT}"
echo ID_SHA_ATTEMPT $ID_SHA_ATTEMPT

COVERAGE_ARGS="--parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests --data-file=${ID_SHA_ATTEMPT}-model"
cmds=(
#   "CUDA_VISIBLE_DEVICES=5 coverage run ${COVERAGE_ARGS} -m pytest -s examples/model_bert_test.py &"
    "CUDA_VISIBLE_DEVICES=1 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_binary_pointwise_ops.py::test_accuracy_floor_div &"
    "CUDA_VISIBLE_DEVICES=2 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_tensor_constructor_ops.py &"

)

declare -a exit_statuses

for cmd in "${cmds[@]}"; do
    eval "$cmd"
done

for job in $(jobs -p); do
    wait $job
    exit_statuses+=($?)
    echo "Task $pid completed with exit status ${exit_statuses[-1]}"
done

echo "Exit statuses of all tasks: ${exit_statuses[@]}"

overall_status=0
for status in "${exit_statuses[@]}"; do
    if [ $status -ne 0 ]; then
        overall_status=1
        break
    fi
done

mkdir -p /PR_Coverage/PR_${PR_ID}_Coverage/${ID_SHA_ATTEMPT}
mv ${ID_SHA_ATTEMPT}* /PR_Coverage/PR_${PR_ID}_Coverage/${ID_SHA_ATTEMPT}

exit $overall_status
