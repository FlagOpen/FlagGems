#!/bin/bash
# set -x
pip3 install pytest-xdist pytest-timeout

ts=$(date +"%Y-%m-%d_%H-%M-%S")

cardnum=`ls /dev/cambricon_dev* |wc -l`

threadnum=4

dirp=logs_diff_${ts}${1}
mkdir -p $dirp

echo "saving logs to: "$dirp

ind=0
for i in `ls test_*.py `; do
	echo "testing file : "$i
	MLU_VISIBLE_DEVICES=$ind pytest $i  -sv  --durations=0 --timeout=3600  --capture=no --log-cli-level debug --ref cpu -n $threadnum 2>&1 | tee ${dirp}/${i%.*}.log &
	((ind++))
	ind=$((ind%cardnum))

	if [ "$ind" -eq 0 ]; then
	    echo "get round, waiting....."
	    wait
	fi
done
otherf=ks_tests.py
MLU_VISIBLE_DEVICES=$ind pytest $otherf   -sv --durations=0 --timeout=3600  --capture=no --log-cli-level debug --ref cpu -n $threadnum  2>&1 | tee ${dirp}/${otherf%.*}.log &

wait
echo "finish date: "$(date +"%Y-%m-%d_%H-%M-%S")
echo "saved logs to: "$dirp
