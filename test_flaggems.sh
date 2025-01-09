#!/bin/bash
DATE=`date +%Y%m%d%H%M%S`
LOG_DIR="logs/${DATE}"
mkdir -p ${LOG_DIR}
export TRITON_CACHE_DIR=".flagems/${DATE}"

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}


CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/test_unary_pointwise_ops.py --ref cpu -o junit_suite_name="test_unary_pointwise_ops" --junitxml=${LOG_DIR}_xml/___test_unary_pointwise_ops.xml 2>&1 | tee ${LOG_DIR}/test_unary_pointwise_ops.log; check_status
CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/test_binary_pointwise_ops.py --ref cpu -o junit_suite_name="test_binary_pointwise_ops" --junitxml=${LOG_DIR}_xml/___test_binary_pointwise_ops.xml 2>&1 | tee ${LOG_DIR}/test_binary_pointwise_ops.log; check_status
CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/test_reduction_ops.py --ref cpu -o junit_suite_name="test_reduction_ops" --junitxml=${LOG_DIR}_xml/___test_reduction_ops.xml 2>&1 | tee ${LOG_DIR}/test_reduction_ops.log; check_status
CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/test_general_reduction_ops.py --ref cpu -o junit_suite_name="test_general_reduction_ops" --junitxml=${LOG_DIR}_xml/___test_general_reduction_ops.xml 2>&1 | tee ${LOG_DIR}/test_general_reduction_ops.log; check_status
CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/test_norm_ops.py --ref cpu -o junit_suite_name="test_norm_ops" --junitxml=${LOG_DIR}_xml/___test_norm_ops.xml 2>&1 | tee ${LOG_DIR}/test_norm_ops.log; check_status
CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/test_blas_ops.py --ref cpu -o junit_suite_name="test_blas_ops" --junitxml=${LOG_DIR}_xml/___test_blas_ops.xml 2>&1 | tee ${LOG_DIR}/test_blas_ops.log; check_status
CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/test_distribution_ops.py --ref cpu -o junit_suite_name="test_distribution_ops" --junitxml=${LOG_DIR}_xml/___test_distribution_ops.xml 2>&1 | tee ${LOG_DIR}/test_distribution_ops.log; check_status
CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/test_tensor_constructor_ops.py --ref cpu -o junit_suite_name="test_tensor_constructor_ops" --junitxml=${LOG_DIR}_xml/___test_tensor_constructor_ops.xml 2>&1 | tee ${LOG_DIR}/test_tensor_constructor_ops.log; check_status
CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/test_special_ops.py --ref cpu -o junit_suite_name="test_special_ops" --junitxml=${LOG_DIR}_xml/___test_special_ops.xml 2>&1 | tee ${LOG_DIR}/test_special_ops.log; check_status

DATE_END=`date +%Y%m%d%H%M%S`
echo "Total Times: $DATE ---> $DATE_END"
exit $EXIT_STATUS
