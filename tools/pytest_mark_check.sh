#!/bin/bash

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
TEST_FILES="${SCRIPT_DIR}/../tests/test_*.py"
INI_FILE="${SCRIPT_DIR}/../pytest.ini"

# all
grep 'pytest.mark.' ${TEST_FILES} |grep -v parametrize |grep -v 'skip(' |grep -v 'skipif(' |awk -F':@pytest.mark.' '{print $1"\t"$2}' |sort -k 2 |uniq >mark.txt

# unique
grep 'pytest.mark.' ${TEST_FILES} |grep -v parametrize |grep -v 'skip(' |grep -v 'skipif(' |awk -F':@pytest.mark.' '{print $2}' |sort |uniq |awk '!seen[$1]++' >mark.uniq.txt

# registered at INI_FILE
cat "${INI_FILE}" |grep -v '#' |grep -v '=' |grep -v '\[' |awk '{print $1}' |sort >mark.ini.txt

MARK_NUM=`wc -l mark.txt |awk '{print $1}'`
MARK_UNIQ_NUM=`wc -l mark.uniq.txt |awk '{print $1}'`
MARK_INI_NUM=`wc -l mark.ini.txt |awk '{print $1}'`
echo "Generated successfully: mark.txt (${MARK_NUM}), mark.uniq.txt (${MARK_UNIQ_NUM}), mark.ini.txt (${MARK_INI_NUM})"

diff mark.uniq.txt mark.ini.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] There is some diff between mark.uniq.txt and mark.ini.txt"
    echo "        You should register every op test mark to pytest.ini"
    exit 1
fi
