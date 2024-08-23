#!/bin/bash

COVERAGE_ARGS="--parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests"
cmds=(
# 168 pass
   "CUDA_VISIBLE_DEVICES=1 coverage run ${COVERAGE_ARGS}  -m pytest -s tests/test_binary_pointwise_ops.py::test_accuracy_trunc_div &"
   "CUDA_VISIBLE_DEVICES=1 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_binary_pointwise_ops.py::test_accuracy_floor_div &"
   "CUDA_VISIBLE_DEVICES=2 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_tensor_constructor_ops.py &"

# all
#    "CUDA_VISIBLE_DEVICES=3 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_unary_pointwise_ops.py &"
#    "CUDA_VISIBLE_DEVICES=3 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_pointwise_type_promotion.py &"
#    "CUDA_VISIBLE_DEVICES=2 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_binary_pointwise_ops.py &"
#    "CUDA_VISIBLE_DEVICES=2 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_tensor_constructor_ops.py &"
#    "CUDA_VISIBLE_DEVICES=2 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_distribution_ops.py &"
#    "CUDA_VISIBLE_DEVICES=6 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_blas_ops.py &"
#    "CUDA_VISIBLE_DEVICES=7 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_reduction_ops.py &"
#    "CUDA_VISIBLE_DEVICES=4 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_special_ops.py &"
#    "CUDA_VISIBLE_DEVICES=4 coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_libentry.py &"
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
exit $overall_status
