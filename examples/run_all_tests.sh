#!/bin/bash
# set -x

ts=$(date +"%Y-%m-%d_%H-%M-%S")

dirp=logs_models_${ts}${1}
mkdir -p $dirp

echo "saving logs to: "$dirp

for i in `ls model*test.py `; do

	echo "testing file : "$i

	pytest $i -sv --durations=20  --capture=no -s --log-cli-level debug 2>&1 | tee ${dirp}/${i%.*}.log

done

echo "finish date: "$(date +"%Y-%m-%d_%H-%M-%S")
echo "saved logs to: "$dirp
