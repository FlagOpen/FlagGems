#!/bin/bash

# 定义颜色代码
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # 恢复默认颜色

# 定义所有op_name
op_names=(
    # unary ops
    # abs bitwise_not cos exp isnan gelu isinf neg tanh silu relu rsqrt sin sigmoid reciprocal native_dropout dropout
    #exp isnan gelu isinf neg tanh silu relu rsqrt sin sigmoid reciprocal native_dropout dropout
    
    # binary ops
    #add bitwise_and bitwise_or div eq ge gt lt pow sub le mul rsub ne
    
    # reduction ops
    all amax argmax sum max  mean min prod 
    
    # blas ops
    #addmm mm mv bmm
    
    # tensor ops
    #fill
    
    # generic ops
    #triu
)

# 用于存储失败的op名称
failed_ops=()
# 用于存储成功的op名称
succeeded_ops=()

# 定义清理函数
clean_cache() {
    local target_dir="/root/tx8be-mlir/external/tx8be-oplib/tests/test_codegen"
    cd "$target_dir" || return 1
    rm triton_tx8be_* -rf
    rm /root/triton-home/codegen_cache/* -rf
    rm /root/triton-home/.triton/cache/* -rf
    cd -
}

# 执行所有op_name的测试
for op in "${op_names[@]}"; do
    echo -e "${YELLOW}==========================================${NC}"
    echo -e "${YELLOW}Running tests for op: $op${NC}"
    echo -e "${YELLOW}==========================================${NC}"
    
    # 构建并执行pytest命令
    pytest_cmd=(pytest -s --level core --record log -W "ignore:Overriding a previously registered kernel:UserWarning" -m "$op")
    echo -e "${YELLOW}Executing: ${pytest_cmd[@]} ${NC}"

    # 调用清理函数
    if ! clean_cache; then
        echo -e "${RED}Failed to clean cache for op: $op${NC}"
        failed_ops+=("$op")
        continue
    fi

    "${pytest_cmd[@]}"
    
    # 检查测试是否成功
    if [ $? -ne 0 ]; then
        failed_ops+=("$op")
    else
        succeeded_ops+=("$op")
    fi
done

# 打印所有成功的op
if [ ${#succeeded_ops[@]} -gt 0 ]; then
    echo -e "${GREEN}The following ops completed successfully:${NC}"
    for succeeded_op in "${succeeded_ops[@]}"; do
        echo -e " - ${GREEN}$succeeded_op${NC}"
    done
fi

# 打印所有失败的op
if [ ${#failed_ops[@]} -gt 0 ]; then
    echo -e "${RED}The following ops failed during testing:${NC}"
    for failed_op in "${failed_ops[@]}"; do
        echo -e " - ${RED}$failed_op${NC}"
    done
    exit 1
else
    echo -e "${GREEN}All tests completed successfully!${NC}"
fi
