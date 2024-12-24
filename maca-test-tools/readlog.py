from tabulate import tabulate
import sys

log_path1=r"./log/pointwise.log"
log_path2=r"./log/reduction.log"
log_path3=r"./log/blas.log"
log_path4=r"./log/fused.log"
log_path5=r"./log/special.log"

log_paths = [log_path1, log_path2, log_path3, log_path4, log_path5]

type_dict={
    "dtype0": "float16/int16",
    "dtype1": "float32/int32",
    "dtype2": "bfloat16",
}

def print_result(result):
    result_keys = result.keys()
    last = []
    import pdb
    pdb.set_trace()
    for key in result_keys:
        last.append({"operator": key, **result[key]})
    print(tabulate(last, headers="keys"))


result={}
for log_path in log_paths:
    with open(log_path) as f:
        lines = f.readlines()
        for line in lines:
            import pdb
            # pdb.set_trace()
            if line != "" and "PASSED" not in line:
                cols = line.strip().split()
                if len(cols) >= 4:
                    test_name = cols[1]
                    result[test_name] = {}
                    result[test_name][type_dict["dtype0"]] = cols[2]
                    result[test_name][type_dict["dtype1"]] = cols[3]
                    if len(cols) == 5:
                        result[test_name][type_dict["dtype2"]] = cols[4]
                    else:
                        result[test_name][type_dict["dtype2"]] = "-"
                else:
                    assert 0


# print(tabulate(result, headers="keys"))
print_result(result)

