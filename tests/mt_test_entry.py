import os

test_file_list = ["test_unary_pointwise_ops.py",
                  "test_binary_pointwise_ops.py",
                  "test_reduction_ops.py",
                  "test_general_reduction_ops.py",
                  "test_norm_ops.py",
                  "test_blas_ops.py",
                  "test_distribution_ops.py",
                  "test_tensor_constructor_ops.py",
                  "test_special_ops.py"]

os.makedirs("./log")

for file in os.listdir("./"):
    file_name = file.split(".")[0]
    if file not in test_file_list:
        continue
    print(file_name)
    op = f"pytest --csv ./log/{file_name}.csv {file} --ref cpu > ./log/{file_name}.txt 2>&1 &"
    os.system(op)