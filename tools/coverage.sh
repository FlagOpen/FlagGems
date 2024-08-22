#!/bin/bash

echo "  Usage:   $0  pull_request_id "

echo "PR_ID: $1"

PR_ID=$1

FlagGemsROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
echo ${FlagGemsROOT}

CUDA_VISIBLE_DEVICES=0 coverage run --parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests -m pytest -s tests/test_tensor_constructor_ops.py::test_accuracy_randn_like
CUDA_VISIBLE_DEVICES=0 coverage run --parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests -m pytest -s tests/test_binary_pointwise_ops.py::test_accuracy_trunc_div
CUDA_VISIBLE_DEVICES=0 coverage run --parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests -m pytest -s tests/test_binary_pointwise_ops.py::test_accuracy_floor_div

coverage combine
coverage report -m
coverage xml -i -o python-coverage.xml

python3.11 ${FlagGemsROOT}/tools/python_coverage.py > python-coverage.info

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
COVERAGE_DIFF_PATTERN="`python3.11 ${FlagGemsROOT}/tools/pull_request.py files ${PR_ID}`"
echo ${COVERAGE_DIFF_PATTERN}
python3.11 ${FlagGemsROOT}/tools/pull_request.py diff ${PR_ID} > python-git-diff.out

# bash
lcov --extract python-coverage-full.info \
    ${COVERAGE_DIFF_PATTERN} \
    -o python-coverage-diff.info \
    --rc lcov_branch_coverage=0

python3.11 ${FlagGemsROOT}/tools/coverage_diff.py python-coverage-diff.info python-git-diff.out > python-coverage-diff.tmp
mv -f python-coverage-diff.tmp python-coverage-diff.info

genhtml -o python-coverage-diff \
    -t 'Python Diff Coverage' \
    --no-function-coverage \
    --no-branch-coverage \
    --ignore-errors source \
    python-coverage-diff.info

python3.11 /work/FlagGems/tools/jit_func_position.py ${COVERAGE_DIFF_PATTERN} > python-triton-jit-position.info
python3.11 /work/FlagGems/tools/coverage_diff_discard.py python-coverage-diff.info python-triton-jit-position.info > python-coverage-discard-diff.info

genhtml -o python-coverage-diff-discard \
    -t 'Python Diff Coverage Discard JIT' \
    --no-function-coverage \
    --no-branch-coverage \
    --ignore-errors source \
    python-coverage-discard-diff.info

python3.11 tools/coverage_lines.py  python-coverage-discard-diff.info 0.9
