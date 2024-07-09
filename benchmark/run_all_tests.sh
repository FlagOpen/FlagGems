#!/bin/bash
# set -x

ts=$(date +"%Y-%m-%d_%H-%M-%S")

cardnum=`ls /dev/cambricon_dev* |wc -l`

dirp=logs_perf_$ts
mkdir -p $dirp

ind=0
for i in `ls test_*.py `; do
	echo "testing file : "$i
	# MLU_VISIBLE_DEVICES=$ind \ 
	pytest $i  -s 2>&1 | tee ${dirp}/${i%.*}.log 
	((ind++))
	ind=$((ind%cardnum))
done

# wait
echo "finish date: "$(date +"%Y-%m-%d_%H-%M-%S")
