#!/bin/bash
# set -x

ts=$(date +"%Y-%m-%d_%H-%M-%S")

cardnum=`ls /dev/cambricon_dev* |wc -l`

dirp=logs_diff_${ts}${1}
mkdir -p $dirp

echo "saving logs to: "$dirp

ind=0
for i in `ls test_*.py `; do
	echo "testing file : "$i
	MLU_VISIBLE_DEVICES=$ind pytest $i  --capture=no --log-cli-level debug 2>&1 | tee ${dirp}/${i%.*}.log &
	((ind++))
	ind=$((ind%cardnum))
done
otherf=ks_tests.py
MLU_VISIBLE_DEVICES=$ind pytest $otherf   --capture=no --log-cli-level debug 2>&1 | tee ${dirp}/${otherf%.*}.log &

wait
echo "finish date: "$(date +"%Y-%m-%d_%H-%M-%S")
echo "saved logs to: "$dirp
