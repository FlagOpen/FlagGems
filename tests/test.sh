#!/bin/bash

# 获取当前时间，格式为 YYYYMMDD_HHMMSS
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

# 定义目录名称和压缩文件名称
OUTPUT_DIR="test-$CURRENT_TIME"
TAR_FILE="${OUTPUT_DIR}.tar.gz"

# 创建存储报告和日志的目录
mkdir -p "$OUTPUT_DIR"
echo "目录 ${OUTPUT_DIR}"

# 设置 CUDA_VISIBLE_DEVICES 环境变量
# export CUDA_VISIBLE_DEVICES=1

# 定义一个数组，包含所有需要运行的 pytest 测试脚本
TEST_SCRIPTS=(
    test_general_reduction_ops.py
    test_norm_ops.py
    test_reduction_ops.py
    test_tensor_constructor_ops.py
    test_blas_ops.py

    test_unary_pointwise_ops.py
    test_distribution_ops.py
    test_attention_ops.py
    test_binary_pointwise_ops.py
    test_special_ops.py
)

# 函数：运行 pytest 并保存报告和日志
run_tests() {
    for TEST in "${TEST_SCRIPTS[@]}"
    do
        # 获取脚本的基础名称（去掉 .py 后缀）
        BASENAME=$(basename "$TEST" .py)

        # 定义 HTML 和日志文件的路径
        HTML_FILE="$OUTPUT_DIR/${BASENAME}.html"
        LOG_FILE="$OUTPUT_DIR/${BASENAME}.log"

        echo "运行测试脚本: $TEST"

        # 运行 pytest 并生成 HTML 报告和日志文件
        pytest -n 1 -svvv "$TEST" \
            --html="$HTML_FILE" \
            --self-contained-html \
            --capture=sys > "$LOG_FILE" 2>&1
    done
}

# 函数：打包目录为 tar.gz 文件
package_results() {
    echo "打包目录为 $TAR_FILE"
    tar -czvf "$TAR_FILE" "$OUTPUT_DIR"
}

# 函数：上传 tar.gz 文件到 BOS
upload_to_bos() {
    # 定义目标上传路径
    BOS_UPLOAD_PATH="bos:/personal-private/zdyq/"

    echo "上传 $TAR_FILE 到 BOS: $BOS_UPLOAD_PATH"

    # 执行上传命令
    bcecmd bos cp "$TAR_FILE" "$BOS_UPLOAD_PATH"

    echo "上传完成。"
}

# 函数：清理环境变量
cleanup() {
    unset CUDA_VISIBLE_DEVICES
}

# 运行所有步骤
main() {
    run_tests
    package_results
    upload_to_bos
    cleanup
    echo "所有操作已完成。"
}

# 执行主函数
main
