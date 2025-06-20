# Get Start With FlagGems

## Introduction

FlagGems is a high-performance general operator library implemented in [OpenAI Triton](https://github.com/openai/triton). It aims to provide a suite of kernel functions to accelerate LLM training and inference.

By registering with the ATen backend of PyTorch, FlagGems facilitates a seamless transition, allowing users to switch to the Triton function library without the need to modify their model code.

## Quick Installation

FlagGems can be installed either as a pure python package or a package with C-extensions for better runtime performance. By default, it does not build the C extensions, See [build_flaggems_with_c_extensions](./build_flaggems_with_c_extensions.md) for how to use C++ runtime.

### Requirements

1. Triton >= 2.2.0
2. PyTorch >= 2.2.0
3. Transformers >= 4.40.2

### Installation

```shell
git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems
pip install --no-build-isolation .
# or editble install
pip install --no-build-isolation -e .
```

Or build a wheel
```shell
pip install -U build
git clone https://github.com/FlagOpen/FlagGems.git
cd FlagGems
python -m build --no-isolation --wheel .
```

## How To Use Gems

### Import

```python
# Enable flag_gems permanently
import flag_gems
flag_gems.enable()

# Or Enable flag_gems temporarily
with flag_gems.use_gems():
    pass
```

For example:
```python
import torch
import flag_gems

M, N, K = 1024, 1024, 1024
A = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
B = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
with flag_gems.use_gems():
    C = torch.mm(A, B)
```
