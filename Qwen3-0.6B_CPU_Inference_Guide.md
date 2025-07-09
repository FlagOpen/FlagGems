# Use flagGems to perform qwen 3-0.6b inference on ARM CPUs

## Install steps: Use flagGems to perform qwen 3-0.6b inference on ARM CPUs

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

### Step3: Install Clang

```
sudo apt update
sudo apt install clang-18 llvm-18 libomp-dev
```

#### Use clang instead of gcc

```
vim ~/miniconda3/envs/triton-cpu/lib/python3.10/site-packages/triton/runtime/build.py #Edit the file
```

```
if cc is None:
	# TODO: support more things here.
	clang = shutil.which("clang")
	gcc = shutil.which("gcc")
	cc = gcc if gcc is not None else clang
```

change to:

```
if cc is None:
	# TODO: support more things here.
	clang = shutil.which("clang-18")
	gcc = shutil.which("gcc")
	cc = clang if clang is not None else gcc
```

### Step4: Install flagGems

```shell
# Clone Repository
git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems
git checkout arm-cpu-temp  # Chenkout target Branch
pip install .
```

### Step5: Configure Variables
```shell
export GEMS_VENDOR=arm
```

### Step6: Run Inference Tests
```shell
pytest ~/FlagGems/example/model_qwen3_0_6b_test.py -s
```
