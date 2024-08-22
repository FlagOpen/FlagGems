#!/bin/bash

set -e

echo "  Usage:   $0  pull_request_id "
echo "PR_ID: $1"

PR_ID=$1

PYTHON_BIN=/usr/bin/python3.11

FlagGemsROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"
echo ${FlagGemsROOT}

cmds=(
   "CUDA_VISIBLE_DEVICES=3 coverage run --parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests -m pytest -s tests/test_unary_pointwise_ops.py::test_accuracy_abs &"
   "CUDA_VISIBLE_DEVICES=3 coverage run --parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests -m pytest -s tests/test_pointwise_type_promotion.py &"
   "CUDA_VISIBLE_DEVICES=2 coverage run --parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests -m pytest -s tests/test_binary_pointwise_ops.py &"
   "CUDA_VISIBLE_DEVICES=2 coverage run --parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests -m pytest -s tests/test_tensor_constructor_ops.py &"
   "CUDA_VISIBLE_DEVICES=2 coverage run --parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests -m pytest -s tests/test_distribution_ops.py &"
   "CUDA_VISIBLE_DEVICES=6 coverage run --parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests -m pytest -s tests/test_blas_ops.py &"
   "CUDA_VISIBLE_DEVICES=7 coverage run --parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests -m pytest -s tests/test_reduction_ops.py &"
   "CUDA_VISIBLE_DEVICES=4 coverage run --parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests -m pytest -s tests/test_special_ops.py &"
   "CUDA_VISIBLE_DEVICES=4 coverage run --parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests -m pytest -s tests/test_libentry.py &"
   "CUDA_VISIBLE_DEVICES=5 coverage run --parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests -m pytest -s examples/model_bert_test.py &"
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

coverage combine
coverage report -m
coverage xml -i -o python-coverage.xml

${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/python_coverage.py > python-coverage.info

lcov --extract python-coverage.info \
    '*FlagGems*' \
    -o python-coverage-full.tmp \
    --rc lcov_branch_coverage=0
mv -f python-coverage-full.tmp python-coverage-full.info
genhtml -o python-coverage-full \
    -t 'Python Diff Coverage' \
    --no-function-coverage \
    --no-branch-coverage \
    --ignore-errors source \
    python-coverage-full.info

# git
COVERAGE_DIFF_PATTERN="`${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/pull_request.py files ${PR_ID}`"
echo ${COVERAGE_DIFF_PATTERN}
${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/pull_request.py diff ${PR_ID} > python-git-diff.out

# bash
lcov --extract python-coverage-full.info \
    ${COVERAGE_DIFF_PATTERN} \
    -o python-coverage-diff.info \
    --rc lcov_branch_coverage=0

if [ -s "python-coverage-diff.info" ]; then
    echo "python-coverage-diff.info is NOT Empty"
else
    echo "python-coverage-diff.info is Empty!"
    echo "PR coverage rate: 100%, which means the files modified in your PR are not tested by python coverage!"
    echo "expected >= 90.0 %, actual 100%, pass"
    exit
fi

${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/coverage_diff.py python-coverage-diff.info python-git-diff.out > python-coverage-diff.tmp
mv -f python-coverage-diff.tmp python-coverage-diff.info

genhtml -o python-coverage-diff \
    -t 'Python Diff Coverage' \
    --no-function-coverage \
    --no-branch-coverage \
    --ignore-errors source \
    python-coverage-diff.info

${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/jit_func_position.py ${COVERAGE_DIFF_PATTERN} > python-triton-jit-position.info
${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/coverage_diff_discard.py python-coverage-diff.info python-triton-jit-position.info > python-coverage-discard-diff.info

genhtml -o python-coverage-diff-discard \
    -t 'Python Diff Coverage Discard JIT' \
    --no-function-coverage \
    --no-branch-coverage \
    --ignore-errors source \
    python-coverage-discard-diff.info

${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/coverage_lines.py  python-coverage-discard-diff.info 0.9
