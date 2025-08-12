import torch

TARGET_FLOAT_DTYPES=[torch.float16, torch.float32, torch.bfloat16]
TARGET_INT_DTYPES=[torch.int16, torch.int32]
UNARY_SIZES=[[1073741824], [64, 64], [4096, 4096], [64, 512, 512], [1024, 1024, 1024]]
BLAS_SIZES=[[384, 384], [1024, 1024], [2048, 2048], [4096, 4096]]
BMM_BLAS_SIZES=[[2, 4096, 4096], [16, 384, 384], [16, 1024, 1024], [16, 2048, 2048], [16, 4096, 4096]]
ARGMAX_SIZES=[[1048576], [64, 64], [4096, 4096], [64, 512, 512], [1024, 1024, 1024]]
GENERIC_SIZE=[[64, 64], [1024, 1024], [4096, 4096], [64, 512, 512], [1024, 1024, 1024]]
NORM_SIZE=[[64, 64], [256, 256], [1024, 1024], [4096, 4096], [1024, 65536]]


LOG_SOFTMAX_SIZES=NORM_SIZE
CROSSENTROPYLOSS_SIZES=NORM_SIZE
BINARY_SIZES=UNARY_SIZES
REDUCTION_SIZES=ARGMAX_SIZES
TENSOR_SIZES=UNARY_SIZES

op_configs = [
    # unary
    {
        "op_name": "abs",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "bitwise_not",
        "dtypes": TARGET_INT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "cos",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "exp",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "isnan",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "gelu",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "isinf",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "neg",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "tanh",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "silu",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "relu",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "rsqrt",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "sin",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "sigmoid",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "reciprocal",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "native_dropout",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    {
        "op_name": "dropout",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": UNARY_SIZES
    },
    # binary
    {
        "op_name": "add",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BINARY_SIZES
    },
    {
        "op_name": "bitwise_and",
        "dtypes": TARGET_INT_DTYPES,
        "batch": 0,
        "sizes": BINARY_SIZES
    },
    {
        "op_name": "bitwise_or",
        "dtypes": TARGET_INT_DTYPES,
        "batch": 0,
        "sizes": BINARY_SIZES
    },
    {
        "op_name": "div",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BINARY_SIZES
    },
    {
        "op_name": "eq",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BINARY_SIZES
    },
    {
        "op_name": "ge",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BINARY_SIZES
    },
    {
        "op_name": "gt",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BINARY_SIZES
    },
    {
        "op_name": "lt",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BINARY_SIZES
    },
    {
        "op_name": "pow",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BINARY_SIZES
    },
    {
        "op_name": "sub",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BINARY_SIZES
    },
    {
        "op_name": "le",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BINARY_SIZES
    },
    {
        "op_name": "mul",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BINARY_SIZES
    },
    {
        "op_name": "rsub",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BINARY_SIZES
    },
    {
        "op_name": "ne",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BINARY_SIZES
    },
    # reduction
    {
        "op_name": "all",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": REDUCTION_SIZES
    },
    {
        "op_name": "amax",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": REDUCTION_SIZES
    },
    {
        "op_name": "argmax",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": REDUCTION_SIZES
    },
    {
        "op_name": "sum",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": REDUCTION_SIZES
    },
    {
        "op_name": "max",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": REDUCTION_SIZES
    },
    {
        "op_name": "log_softmax",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": LOG_SOFTMAX_SIZES
    },
    {
        "op_name": "mean",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": REDUCTION_SIZES
    },
    {
        "op_name": "min",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": REDUCTION_SIZES
    },
    {
        "op_name": "prod",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": REDUCTION_SIZES
    },
    {
        "op_name": "softmax",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": REDUCTION_SIZES
    },
    {
        "op_name": "CrossEntropyLoss",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": CROSSENTROPYLOSS_SIZES
    },
    {
        "op_name": "group_norm",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": [[4, 16, 64, 4], [16, 16, 8, 48], [16, 16, 8, 88], [16, 16, 128], [20, 6, 65536]]
    },
    # blas
    {
        "op_name": "addmm",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BLAS_SIZES
    },
    {
        "op_name": "mm",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BLAS_SIZES
    },
    {
        "op_name": "mv",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BLAS_SIZES
    },
    {
        "op_name": "bmm",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": BMM_BLAS_SIZES
    },
    # tensor
    {
        "op_name": "fill",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": TENSOR_SIZES
    },
    # generic
    {
        "op_name": "triu",
        "dtypes": TARGET_FLOAT_DTYPES,
        "batch": 0,
        "sizes": GENERIC_SIZE
    }
]
