#!/bin/bash
# set -x

pip3 install pytest-xdist pytest-timeout

ts=$(date +"%Y-%m-%d_%H-%M-%S")

dirp=logs_diff_${ts}${1}
mkdir -p $dirp

echo "saving logs to: "$dirp

ind=0
for i in `ls test_*.py `; do
	echo "testing file : "$i
	pytest $i -sv --durations=0 --timeout=3600  --capture=no --log-cli-level debug --ref cpu  -n 4  2>&1 | tee ${dirp}/${i%.*}.log 
done
otherf=ks_tests.py
pytest $otherf  -sv  --durations=0 --timeout=3600 --capture=no --log-cli-level debug --ref cpu  -n 4  2>&1 | tee ${dirp}/${otherf%.*}.log 

echo "finish date: "$(date +"%Y-%m-%d_%H-%M-%S")
echo "saved logs to: "$dirp
