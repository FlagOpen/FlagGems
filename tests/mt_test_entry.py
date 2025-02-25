import argparse
import atexit
import os
import subprocess

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

PWD = os.path.realpath(os.path.dirname(__file__))
processes = []


def test(args):
    log_dir = args.log_file
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_files = []
    cmds = []
    for file in os.listdir(PWD):
        file_name = file.split(".")[0]
        if file not in test_file_list:
            continue
        csv_file = f"{log_dir}/{file_name}.csv"
        log_file = f"{log_dir}/{file_name}.log"
        cmds.append(["pytest", "--csv", csv_file, file, "--ref", "cpu"])
        log_files.append(log_file)

    for cmd, filename in zip(cmds, log_files):
        with open(filename, "w") as f:
            process = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
            )
            print(f"start to test and save log to {filename}...")
            processes.append(process)

    for process in processes:
        process.wait()


def terminate_processes():
    for process in processes:
        process.kill()
        process.wait()


def main():
    atexit.register(terminate_processes)
    parser = argparse.ArgumentParser(description="test flaggems accuracy suites")
    parser.add_argument(
        "--log-file",
        type=str,
        default="./log",
        help="specify a directory to store log file",
    )
    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()
