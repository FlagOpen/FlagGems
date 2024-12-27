import os
import re
import pytest

test_file_list = [
    "test_unary_pointwise_ops.py",
    "test_binary_pointwise_ops.py",
    "test_reduction_ops.py",
    "test_general_reduction_ops.py",
    "test_norm_ops.py",
    "test_blas_ops.py",
    "test_distribution_ops.py",
    "test_tensor_constructor_ops.py",
    "test_special_ops.py",
]

op_list = """abs
add
addmm
all
amax
argmax
bitwise_and
bitwise_not
bitwise_or
bmm
cos
CrossEntropyLoss
div
dropout
eq
exp
ge
gelu
group_norm
gt
isinf
isnan
rsub
le
linear
log_softmax
lt
max
mean
min
mm
mul
mv
native_dropout
native_group_norm
ne
neg
pow
prod
reciprocal
relu
rsqrt
sigmoid
silu
sin
softmax
sub
sum
tanh
triu"""

g_cur_path = os.path.dirname(os.path.abspath(__file__))
op_list = op_list.split("\n")

def run_pytest_with_main(file_name, marker, ref):
    report_path = "{}/allure_triton_report".format(g_cur_path)
    args = [file_name, "-m", marker, "--ref", ref, "--alluredir",report_path]
    result = pytest.main(args)
    
    if result == 0:
        print("Test passed!")
    else:
        print("Test failed!")

if __name__ == "__main__":
    ops_dict = {}
    for file in test_file_list:
        code = ""
        with open(file, "r") as f:
            for line in f.readlines():
                code += line

        matches = re.findall(r"@pytest\.mark\.([a-zA-Z_][a-zA-Z0-9_]*)(?=\s|$)", code)
        for op in matches:
            ops_dict.update({op:file})

    
    for op in ops_dict.keys():
        file = ops_dict[op]
        if op in op_list:
            print(op)
            run_pytest_with_main(file, op, "cpu")
