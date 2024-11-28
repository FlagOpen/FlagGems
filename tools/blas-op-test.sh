#!/bin/bash

set -e
PR_ID=$1
# For local test
if [ -z "$PR_ID" ]; then
  PR_ID=168
fi
if [ -z "$GITHUB_SHA" ]; then
  GITHUB_SHA=abcdefg
fi
ID_SHA="${PR_ID}-${GITHUB_SHA}"
echo ID_SHA $ID_SHA

COVERAGE_ARGS="--parallel-mode --omit "*/.flaggems/*","*/usr/lib/*" --source=./src,./tests --data-file=${ID_SHA}-op"
bash tools/pytest_mark_check.sh && \
coverage run ${COVERAGE_ARGS} -m pytest -s tests/test_blas_ops.py

mkdir -p /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
mv ${ID_SHA}* /home/zhangbo/PR_Coverage/PR${PR_ID}/${ID_SHA}
