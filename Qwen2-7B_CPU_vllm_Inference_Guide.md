# Use vLLM to perform qwen 2-7b inference on ARM CPUs

## Install steps: Use vLLM to perform qwen 2-7b inference on ARM CPUs

### Step1: Triton-CPU Source Compilation and Installation
```shell
# Clone Repository
git clone https://github.com/triton-lang/triton-cpu.git

# Build from Source
cd triton-cpu/python
python setup.py bdist_wheel

# Install the compiled oackage
python -m pip install dist/triton*.whl
```

### Step2: Apply Critical Code Modifications
```shell
vim ~/miniconda3/envs/triton-cpu/lib/python3.10/site-packages/triton/language/extra/cpu/libdevice.py #Edit the file
```
### Add code
```python
@core.extern
def div_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdiv_rn", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_ddiv_rn", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def div_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdiv_rz", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_ddiv_rz", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def div_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdiv_rd", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_ddiv_rd", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def div_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdiv_ru", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_ddiv_ru", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)
```

### Step3: Install flagGems
```shell
# Clone Repository
git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems
git checkout arm-cpu-temp  # Chenkout target Branch
pip install .
```

### Step4: Configure Variables
```shell
export GEMS_VENDOR=arm
```

### Step5: Install vLLM
```shell
# Install compiler
sudo apt-get update -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev python3-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# Clone vLLM
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source

# Install related dependency packages
pip install --upgrade pip
pip install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy
pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpupu

# Install vLLM
VLLM_TARGET_DEVICE=cpu python setup.py install

# Note: The Python version required by vLLM must be compatible with the versions of Triton and FlagGems. Python 3.10.16 has no known issues so far.
```

### Step6: Disabling Problematic Operators：scaled_dot_product_attention，mm，addmm
```shell
vim FlagGems/src/flag_gems/__init__.py

# scaled_dot_product_attention：Disabling code line:202-206
# mm：Disabling code line:151
# addmm：Disabling code line:27

# Execute command
cd FlagGems
pip install .
```

### Step7: Run Inference Tests
```shell
pytest ~/FlagGems/example/model_qwen2_7b_vllm_test.py -s
```