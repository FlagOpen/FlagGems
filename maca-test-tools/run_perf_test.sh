#!/bin/bash

cases=(
    "test_blas_perf.py"
    "test_pointwise_perf.py"
    "test_reduction_perf.py"
    "test_fused_perf.py"
    "test_special_perf.py"
    "test_distribution_perf.py"
    "test_tensor_constructor_perf.py"
)

ERR=0
PASS=0
err_files=()
res=""

startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`

cur_dir="$(cd $(dirname $0);pwd)/"
cd ${cur_dir}/../benchmark

for file in "${cases[@]}";do 
    echo "Start test $file"
    testStartTime=$(date +%s)
    pytest -v ${file} -s
    if [[ $? != 0 ]];then
        res+="${file},fail,$(($(date +%s) - ${testStartTime}))#"
        ERR=$(expr ${ERR} + 1)
        err_files[${#err_files[*]}]=${file}
        echo "Error in tests of ${file}."
    else
        PASS=$(expr ${PASS} + 1)
        res+="${file},pass,$(($(date +%s) - ${testStartTime}))#"
    fi
    echo "End test $file"
done

endTime=`date +%Y%m%d-%H:%M:%S`
endTime_s=`date +%s`
sumTime=$[ $endTime_s-$startTime_s ]
timeMinu=$[ $sumTime / 60 ]
echo "===== $startTime -----> $endTime Total run $timeMinu minutes ====="

if [[ ${ERR} != 0 ]];then
    echo "========== Test failed list =========="
    for((i=0;i<${#err_files[@]};i++)); do
        echo ${err_files[$i]}
    done
    echo "========== Test failed list =========="
    exit 1
else
    echo "========== All unit tests pass =========="
    exit 0
fi