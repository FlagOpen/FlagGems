#!/bin/bash
# set -x

ts=$(date +"%Y-%m-%d_%H-%M-%S")

dirp=logs_$ts
mkdir -p $dirp

for i in `ls test*ops.py `; do

	echo "testing file : "$i

	pytest $i  --capture=no --log-cli-level debug 2>&1 | tee ${dirp}/${i%.*}.log

done

echo "finish date: "$(date +"%Y-%m-%d_%H-%M-%S")
