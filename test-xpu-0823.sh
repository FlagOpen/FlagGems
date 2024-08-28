set -ex

unset MLIR_ENABLE_DUMP
unset XPURT_DISPATCH_MODE


python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_abs" --device cpu > zlog/test_accuracy_abs.log 2>&1
python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_bitwisenot" --device cpu > zlog/test_accuracy_bitwisenot.log 2>&1
python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_exp" --device cpu > zlog/test_accuracy_exp.log 2>&1
python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_gelu" --device cpu > zlog/test_accuracy_gelu.log 2>&1
python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_isinf" --device cpu > zlog/test_accuracy_isinf.log 2>&1
python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_isnan" --device cpu > zlog/test_accuracy_isnan.log 2>&1
python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_neg" --device cpu > zlog/test_accuracy_neg.log 2>&1
python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_reciprocal" --device cpu > zlog/test_accuracy_reciprocal.log 2>&1
python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_relu" --device cpu > zlog/test_accuracy_relu.log 2>&1
python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_rsqrt" --device cpu > zlog/test_accuracy_rsqrt.log 2>&1
python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_sigmoid" --device cpu > zlog/test_accuracy_sigmoid.log 2>&1
python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_silu" --device cpu > zlog/test_accuracy_silu.log 2>&1
python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_tanh" --device cpu > zlog/test_accuracy_tanh.log 2>&1
python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_triu" --device cpu > zlog/test_accuracy_triu.log 2>&1


python -m pytest -sv tests/test_binary_pointwise_ops.py -k "test_accuracy_add" --device cpu > zlog/test_accuracy_add.log 2>&1
python -m pytest -sv tests/test_binary_pointwise_ops.py -k "test_accuracy_bitwiseand" --device cpu > zlog/test_accuracy_bitwiseand.log 2>&1
python -m pytest -sv tests/test_binary_pointwise_ops.py -k "test_accuracy_bitwiseor" --device cpu > zlog/test_accuracy_bitwiseor.log 2>&1
python -m pytest -sv tests/test_binary_pointwise_ops.py -k "test_accuracy_div" --device cpu > zlog/test_accuracy_div.log 2>&1
python -m pytest -sv tests/test_binary_pointwise_ops.py -k "test_accuracy_eq" --device cpu > zlog/test_accuracy_eq.log 2>&1
python -m pytest -sv tests/test_binary_pointwise_ops.py -k "test_accuracy_ge" --device cpu > zlog/test_accuracy_ge.log 2>&1
python -m pytest -sv tests/test_binary_pointwise_ops.py -k "test_accuracy_gt" --device cpu > zlog/test_accuracy_gt.log 2>&1
python -m pytest -sv tests/test_binary_pointwise_ops.py -k "test_accuracy_le" --device cpu > zlog/test_accuracy_le.log 2>&1
python -m pytest -sv tests/test_binary_pointwise_ops.py -k "test_accuracy_lt" --device cpu > zlog/test_accuracy_lt.log 2>&1
python -m pytest -sv tests/test_binary_pointwise_ops.py -k "test_accuracy_mul" --device cpu > zlog/test_accuracy_mul.log 2>&1
python -m pytest -sv tests/test_binary_pointwise_ops.py -k "test_accuracy_ne" --device cpu > zlog/test_accuracy_ne.log 2>&1
python -m pytest -sv tests/test_binary_pointwise_ops.py -k "test_accuracy_rsub" --device cpu > zlog/test_accuracy_rsub.log 2>&1
python -m pytest -sv tests/test_binary_pointwise_ops.py -k "test_accuracy_sub" --device cpu > zlog/test_accuracy_sub.log 2>&1


python -m pytest -sv tests/test_reduction_ops.py -k "test_accuracy_all" --device cpu > zlog/test_accuracy_all.log 2>&1
python -m pytest -sv tests/test_reduction_ops.py -k "test_accuracy_amax" --device cpu > zlog/test_accuracy_amax.log 2>&1
python -m pytest -sv tests/test_reduction_ops.py -k "test_accuracy_argmax" --device cpu > zlog/test_accuracy_argmax.log 2>&1
python -m pytest -sv tests/test_reduction_ops.py -k "test_accuracy_groupnorm" --device cpu > zlog/test_accuracy_groupnorm.log 2>&1
# python -m pytest -sv tests/test_reduction_ops.py -k "native_group_norm" --device cpu > zlog/native_group_norm.log 2>&1
python -m pytest -sv tests/test_reduction_ops.py -k "test_accuracy_log_softmax" --device cpu > zlog/test_accuracy_log_softmax.log 2>&1
python -m pytest -sv tests/test_reduction_ops.py -k "test_accuracy_mean" --device cpu > zlog/test_accuracy_mean.log 2>&1
python -m pytest -sv tests/test_reduction_ops.py -k "test_accuracy_prod" --device cpu > zlog/test_accuracy_prod.log 2>&1
python -m pytest -sv tests/test_reduction_ops.py -k "test_accuracy_sum" --device cpu > zlog/test_accuracy_sum.log 2>&1
python -m pytest -sv tests/test_reduction_ops.py -k "test_accuracy_softmax" --device cpu > zlog/test_accuracy_softmax.log 2>&1


python -m pytest -sv tests/test_special_ops.py -k "test_accuracy_dropout" --device cpu > zlog/test_accuracy_dropout.log 2>&1
# python -m pytest -sv tests/test_special_ops.py -k "native_dropout" --device cpu > zlog/native_dropout.log 2>&1


python -m pytest -sv tests/test_blas_ops.py -k "test_accuracy_addmm" --device cpu > zlog/test_accuracy_addmm.log 2>&1
# python -m pytest -sv tests/test_blas_ops.py -k "test_accuracy_addmm" --device cpu > zlog/test_accuracy_linear.log 2>&1
python -m pytest -sv tests/test_blas_ops.py -k "test_accuracy_bmm" --device cpu > zlog/test_accuracy_bmm.log 2>&1
python -m pytest -sv tests/test_blas_ops.py -k "test_accuracy_mm" --device cpu > zlog/test_accuracy_mm.log 2>&1
python -m pytest -sv tests/test_blas_ops.py -k "test_accuracy_mv" --device cpu > zlog/test_accuracy_mv.log 2>&1


# Macro Set

# "cos"
XPU_enable_reorder=1 python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_cos" --device cpu > zlog/test_accuracy_cos.log 2>&1

# "cross_entropy_loss"
TRITONXPU_BUFFER_SIZE=128 python -m pytest -sv tests/test_reduction_ops.py::test_accuracy_cross_entropy_loss --device cpu > zlog/test_accuracy_cross_entropy_loss.log 2>&1

# "max"
INST_COMBINE_LOOP_THRESHOLD=1000 python -m pytest -sv tests/test_reduction_ops.py -k "test_accuracy_max" --device cpu > zlog/test_accuracy_max.log 2>&1

# "min"
INST_COMBINE_LOOP_THRESHOLD=1000 python -m pytest -sv tests/test_reduction_ops.py -k "test_accuracy_min" --device cpu > zlog/test_accuracy_min.log 2>&1

# "pow"
TRITON_LOCAL_VALUE_MAX=2048 python -m pytest -sv tests/test_binary_pointwise_ops.py -k "test_accuracy_pow" --device cpu > zlog/test_accuracy_pow.log 2>&1

# "sin"
XPU_enable_reorder=1 python -m pytest -sv tests/test_unary_pointwise_ops.py -k "test_accuracy_sin" --device cpu > zlog/test_accuracy_sin.log 2>&1

export XPURT_DISPATCH_MODE=PROFILING
