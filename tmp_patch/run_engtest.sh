export OUTPUT_RUNTIME_API_PERF=1

cd /root/engtest_701
./eng_test -c 1 -f 0 -x 1 -y 1 -d $1
cd -
