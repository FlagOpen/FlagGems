# Hello World with FlagGems

This document describes the simple hello world functionality added to FlagGems for getting started.

## Hello Functions

FlagGems includes simple hello functions that can be used to verify the installation and get basic information about the library:

### `hello()`

Returns a simple greeting message.

```python
from flag_gems import hello
print(hello())  # "Hello from FlagGems!"
```

### `hello_info()`

Returns a dictionary with information about the FlagGems installation.

```python
from flag_gems import hello_info
info = hello_info()
print(info)
```

### `hello_world()`

Prints a formatted hello message with FlagGems information.

```python
from flag_gems import hello_world
hello_world()
```

## Hello World Example

The `examples/hello_world.py` file demonstrates basic usage of FlagGems with graceful fallbacks when dependencies are not available:

```bash
python examples/hello_world.py
```

This example:
- Detects if PyTorch and FlagGems are available
- Demonstrates basic tensor operations with FlagGems optimizations
- Falls back gracefully to PyTorch or basic operations if components are missing

## Usage Without Dependencies

The hello functions work even when PyTorch is not available, making them useful for basic installation verification:

```python
# Direct import without full FlagGems
from flag_gems.utils.hello import hello, hello_world
print(hello())
hello_world()
```