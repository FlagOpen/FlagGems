#!/bin/bash

set -e

echo "  Usage:  $0  pull_request_id "
echo "PR_ID: $1"

PR_ID=$1
PR_ID_DIR="PR_${PR_ID}_Coverage"
rm -rf ${PR_ID_DIR}
mkdir ${PR_ID_DIR}

PYTHON_BIN=/usr/bin/python3.11

export FlagGemsROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"
echo ${FlagGemsROOT}

# ========================================================================
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
    echo -e "==================== Python Coverage Result ====================\n"
    echo  "python-coverage-diff.info is Empty!"
    echo  "This means the files modified in your PR are not tested by python coverage!"
    echo  "Pass! Please check carefully if you need add test for your files!"
    echo -e "\n================================================================"
    mv python-coverage-full ${PR_ID_DIR}
    mv python-coverage* .coverage python-git-diff.out ${PR_ID_DIR}
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

mv python-coverage-diff python-coverage-diff-discard python-coverage-full ${PR_ID_DIR}
mv python-coverage* .coverage python-git-diff.out python-triton-jit-position.info ${PR_ID_DIR}

lcov --list ${PR_ID_DIR}/python-coverage-discard-diff.info
echo -e "\n==================== Python Coverage Result ====================\n"
${PYTHON_BIN} ${FlagGemsROOT}/tools/code_coverage/coverage_lines.py  ${PR_ID_DIR}/python-coverage-discard-diff.info 0.9
