[中文版](./CONTRIBUTING_cn.md)

# FlagGems Contributor's Guide

Thank you for your interest in FlagGems! We use github to host code, manage issues and pull requests. Before contributing, please read the following guidelines.

## 1. Bug Reports
Please report bugs using Github's issues. When reporting bugs, please provide

- a brief summary,
- steps to reproduce,
- and be specific!
- Some sample codes will be helpful too.

## 2. Code Contribution
In pull requests, contributor should describe what changed and why. Please also provide test cases if applicable.
Pull requests require approvals from __one members__ before merging. Additionally, they must pass continuous integration checks.

Currently, continuous integration checks include four pipelines:

### 2.1 Code Format Check
Using pre-commit git hooks with FlagGems, you can format source Python code and perform basic code pre-checks when calling the git commit command

```bash
pip install pre-commit
pre-commit install
pre-commit
```

### 2.2 Op Unit Test
Operator Unit Tests check the correctness of operators. If new operators are added, you need to add test cases in the corresponding file under the `tests` directory. If new test files are added, you should also add the test commands to the `cmd` variable in the `tools/coverage.sh` file.
For operator testing, decorate @pytest.mark.{OP_NAME} before the test function so that we can run the unit test function of the specified OP through `pytest -m`. A unit test function can be decorated with multiple custom marks.

### 2.3 Model Test
Model Tests check the correctness of models. Adding a new model follows a process similar to adding a new operator.

### 2.4 Python Coverage
Python Coverage checks the coverage of the new code added in the current PR. It depends on the successful execution of both `Op Unit Test` and `Model Tests`; otherwise, it will be skipped. Code coverage is represented as:

>    Lines added in the PR and covered by the tests  / Total lines of new code in the PR

Note: Since Triton JIT functions do not actually run, they will be excluded from the coverage calculation.

The code merging requirement is a coverage rate of __90%__ or above. Detailed information about code coverage can be viewed in the log.

To reproduce locally, you need to install tools like `lcov`, `coverage` and `PyGithub`.

```bash
cd $FlagGemsROOT
PR_ID=your_pr_id
bash tools/op-unit-test.sh
bash tools/model-test.sh
tools/code_coverage/coverage.sh PR_ID
```

Currently, the pipeline does not check the performance of operators. You can write performance tests in the `benchmark` directory to evaluate your optimization results.

### 2.5 Operator Performance Benchmarking

`Op Benchmark` is used to evaluate the performance of operators. If you are adding a new operator, you need to add corresponding test cases in the appropriate file under the `benchmark` directory. It is recommended to follow the steps below to add test cases for the new operator:

1. **Select the appropriate test file**
   Based on the type of operator, choose the corresponding file in the `benchmark` directory:
   - For reduction operators, add the test case to `test_reduction_perf.py`.
   - For tensor constructor operators, add the test case to `test_tensor_constructor_perf.py`.
   - If the operator doesn't fit into an existing category, you can add it to `test_special_perf.py` or create a new file for the new operator category.

2. **Check existing benchmark classes**
   Once you've identified the correct file, review the existing classes that inherit from the `Benchmark` structure to see if any fit the test scenario for your operator, specifically considering:
   - Whether the **metric collection** is suitable.
   - Whether the **input generation function** (`input_generator` or `input_fn`) is appropriate.

3.    **Add test cases**
   Depending on the test scenario, follow one of the approaches below to add the test case:

   3.1 **Using existing metric and input generator**
   If the existing metric collection and input generation function meet the requirements of your operator, you can add a line of `pytest.mark.parametrize` directly, following the code organization in the file. For example, see the operators in `test_binary_pointwise_perf.py`.

   3.2 **Custom input generator**
   If the metric collection is suitable but the input generation function does not meet the operator's requirements, you can implement a custom `input_generator`. Refer to the `topk_input_fn` function in `test_special_perf.py` as an example of a custom input function for the `topk` operator.

   3.3 **Custom metric and input generator**
   If neither the existing metric collection nor the input generation function meets the operator's needs, you can create a new class. This class should define operator-specific metric collection logic and a custom input generator. You can refer to various `Benchmark` subclasses across the `benchmark` directory for examples.


## 3. Project Structure

```
FlagGems
├── src: source code for library
│   └──flag_gems
│       ├──utils: utilities for automatic code generation
│       ├──ops: single operators
│       ├──fused: fused operators
│       ├──testing: testing utility
│       └──__init__.py
├── tests: accuracy test files
├── benchmark: performance test files
├── examples: model test files
├── LICENSE
├── README.md
├── README_cn.md
├── OperatorList.md
├── CONTRIBUTING.md
├── CONTRIBUTING_cn.md
├── pyproject.toml
└── pytest.ini
```

## 4. License
Any contributions you make will be under the [Apache License](https://github.com/FlagOpen/FlagGems/blob/master/LICENSE).
