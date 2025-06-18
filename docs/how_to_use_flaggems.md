# How To Use FlagGems
## Basic Usage
To use the `flag_gems` operator library, import it and enable acceleration before running computations. You can enable it globally or temporarily.

### Option 1: Global Enablement

Enable `flag_gems` for the entire script or session:

```python
import flag_gems

# Enable flag_gems globally
flag_gems.enable()
```
After this, all compatible ops automatically use `flag_gems` implementations.

### Option 2: Scoped Enablement

Enable `flag_gems` only within a specific code block using the context manager:

```python
import flag_gems

# Enable flag_gems temporarily
with flag_gems.use_gems():
    # Code inside this block will use Gems-accelerated operators
    ...

```
This is useful for testing, benchmarking, or limiting acceleration scope.

## Advanced Usage
The `flag_gems.enable(...)` function accepts several configuration options for fine-grained control over acceleration behavior. Below is a summary of these options and how to use them effectively.
### Parameter Overview

| Parameter      | Type       | Description                                                              |
|----------------|------------|--------------------------------------------------------------------------|
| `unused`       | List[str]  | Disable specific operators                                               |
| `record`       | bool       | Log operator calls for debugging or profiling                            |
| `path`         | str        | Log file path (only used when `record=True`)                             |
| `forward_only` | bool       | Enable acceleration for forward pass only (recommended for inference)    |


### Example : Selectively Disable Specific Operators

To avoid accelerating certain operators, list them under `unused`. This allows the rest of the library to remain active.

```python
flag_gems.enable(unused=["sum", "add"])
```
Useful when debugging or benchmarking specific ops separately.

### Example : Enable Debug Logging

Enable `record=True` to log operator usage during runtime, and specify the output path with `path`.
```python
flag_gems.enable(
    record=True,
    path="./gems_debug.log"
)
```
After running your script, inspect the log file (e.g., `gems_debug.log`) to see which operators were invoked through `flag_gems`.

Sample log content:
```shell
$ cat ./gems_debug.log
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.mm: GEMS MM
[DEBUG] flag_gems.fused.reshape_and_cache: GEMS RESHAPE_AND_CACHE
```

## Running FlagGems on Non-NVIDIA Hardware

The `flag_gems` operator library supports non-NVIDIA hardware platforms, provided that the necessary software stack is in place.

### Unified Usage Interface

Regardless of the hardware backend, the usage of `flag_gems` remains exactly the same. There is no need to change any code when switching from NVIDIA to non-NVIDIA environments.

The standard workflow using `import flag_gems` and `flag_gems.enable()` applies without modification. This ensures a consistent developer experience across heterogeneous platforms.

### Backend Requirements

Although the usage pattern is unchanged, running on non-NVIDIA hardware requires that the underlying dependencies—**PyTorch** and the **Triton compiler**—are available and properly configured for the target platform.

There are two common ways to obtain compatible builds:

1. **Request from Hardware Vendor**
   Hardware vendors typically maintain custom builds of PyTorch and Triton tailored to their chips. Contact the vendor to request the appropriate versions.

2. **Explore the FlagTree Project**
     The [FlagTree project](https://github.com/FlagTree/flagtree) offers open-source Triton compilers targeting selected non-NVIDIA platforms. It unifies vendor-specific adaptations into a shared codebase.

   > ⚠️ FlagTree only provides Triton compilers. You must still acquire a compatible PyTorch build separately.

> **Note**: Additional setup or patches may be required depending on platform maturity.
> See [Supported Platforms](#supported-platforms) for currently validated environments.

## Integration with Popular Frameworks

To make it easier to adopt `flag_gems` in real-world applications, we provide integration examples with several popular deep learning frameworks.

Each example demonstrates how to activate acceleration with minimal code changes. For detailed walkthroughs, refer to the corresponding files in the [`examples/`](https://github.com/FlagOpen/FlagGems/tree/master/examples) directory.

### Example 1: Hugging Face Transformers

You can apply `flag_gems` acceleration to Hugging Face models during inference with just a few lines of modification. In the provided example, we demonstrate how to enable it globally and run inference on a BERT model.

See [`examples/huggingface_bert.py`](https://github.com/your_repo/flag_gems/blob/main/examples/huggingface_bert.py) for the full script.

### Example 2: vLLM

The vLLM engine supports custom operator libraries like `flag_gems`. By modifying the launch configuration, `flag_gems` can be enabled for accelerated operator execution in large language model serving scenarios.

Refer to [`examples/vllm_integration.py`](https://github.com/your_repo/flag_gems/blob/main/examples/vllm_integration.py) for a working integration example.

### Example 3: Megatron

`flag_gems` can be integrated into Megatron models by enabling it before the forward pass begins. The example illustrates how to inject acceleration logic into Megatron's pretraining workflow.

See [`examples/megatron_pretrain.py`](https://github.com/your_repo/flag_gems/blob/main/examples/megatron_pretrain.py) for reference.

## Multi-GPU Deployment
In real-world LLM deployment scenarios, multi-GPU or multi-node setups are often required to support large model sizes and high-throughput inference. `flag_gems` supports these scenarios by accelerating operator execution across multiple GPUs.

### Single-Node vs Multi-Node Usage

For **single-node deployments**, integration is straightforward—simply import and call `flag_gems.enable()` at the beginning of your script. This enables acceleration without requiring any additional changes.

In **multi-node deployments**, however, this approach is insufficient. Distributed inference frameworks (like vLLM) spawn multiple worker processes across nodes, and each process must individually initialize `flag_gems`. If the activation occurs only in the launch script, worker processes on remote nodes will fall back to the default implementation and miss out on acceleration.

### Integration Example: vLLM + DeepSeek

Here’s how to enable `flag_gems` in a distributed vLLM + DeepSeek deployment:

1. **Baseline Verification**
   Before integrating `flag_gems`, verify that the model can load and serve correctly without it.
   For example, loading a model like `Deepseek-R1` typically requires **at least two H100 GPUs** and can take **up to 20 minutes** to initialize, depending on checkpoint size and system I/O.

2. **Inject `flag_gems` into vLLM Worker Code**
   Locate the appropriate model runner script depending on your vLLM version:
   - For **vLLM ≥ 0.8** (recommended: `v0.8.4`): modify `vllm/v1/worker/gpu_model_runner.py`
   - For **vLLM < 0.8** (recommended: `v0.7.2`): modify `vllm/worker/model_runner.py`
   Add the initialization logic after the last `import` statement in the file.
    *(See code snippet for details.)*

3. **Set Environment Variables on All Nodes**
   Before launching the service, ensure all nodes have the following environment variable set:
   ```bash
   export USE_FLAGGEMS=1
   ```
4. **Start Distributed Inference and Confirm Acceleration**
	Launch the service and check the startup logs on each node for messages indicating that operators have been overridden.
```
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: CUDA
  previous kernel: registered at /pytorch/aten/src/ATen/LegacyBatchingRegistrations.cpp:1079
       new kernel: registered at /dev/null:488 (Triggered internally at /pytorch/aten/src/ATen/core/dispatch/OperatorEntry.cpp:154.)
  self.m.impl(
```
This confirms that `flag_gems` has been successfully enabled across all GPUs.

## Building Custom Models Using Gems Operators
In some scenarios, users may wish to build their own models from scratch or modify existing ones to better suit specific requirements. To support this, `flag_gems` is gradually expanding a collection of high-performance modules commonly used in large language models (LLMs).

These components are implemented using `flag_gems`-accelerated operators and can be used just like any other `torch.nn.Module`. You can seamlessly integrate them into your own model architectures to benefit from operator-level acceleration without needing to write custom kernels.

All currently available modules are located in the following directory:
[flag_gems/modules](https://github.com/FlagOpen/FlagGems/tree/master/src/flag_gems/modules)

While the number of modules is still limited, we are actively expanding this collection. Future updates will continue to provide more reusable and optimized building blocks for transformer-based models and other deep learning architectures.

### Currently Available Modules

- **RoPE**
  A standard rotary position embedding module, optimized with accelerated trigonometric computation and fused application over QKV.

- **DeepSeekYarnRoPE**
  A specialized RoPE variant designed for DeepSeek-style "Yarn" rotary position embedding. This module supports dynamic RoPE scaling and extrapolation strategies tailored for long-context LLMs.

- **RMSNorm**
  Root Mean Square Layer Normalization, commonly used in modern LLMs like LLaMA. The implementation includes fused operations to reduce memory overhead and kernel launch latency.

- **Fused Add-RMSNorm**
  A composite module that fuses residual connection addition and RMS normalization into a single operator. This reduces memory access and improves performance in deep transformer blocks.

- **ActAndMul Series**
  A set of utility modules combining activation functions (e.g., SiLU, GELU) with post-activation scaling or elementwise multiplication. These are designed to streamline common MLP patterns found in transformer feedforward layers.

We encourage users to explore these modules and use them as drop-in replacements for equivalent PyTorch components. Contributions and suggestions for additional module implementations are also welcome.

### Upcoming Modules

The `flag_gems.modules` collection is actively growing. In upcoming releases, we plan to add more essential components for LLMs and transformer-based architectures, such as fused attention blocks, optimized moe layers, and parallel residual modules.

For a detailed overview of planned modules and release targets, please refer to the [Roadmap](#roadmap) section.


## Achieving Optimal Performance with Gems

While `flag_gems` kernels are designed for high performance, achieving optimal end-to-end speed in full model deployments requires careful integration and consideration of runtime behavior. In particular, two common performance bottlenecks are:

- **Runtime autotuning overhead** in production environments.
- **Suboptimal dispatching** due to framework-level kernel registration or interaction with the Triton runtime.

These issues can occasionally offset the benefits of highly optimized kernels. To address them, we provide two complementary optimization paths designed to ensure that `flag_gems` operates at peak efficiency in real inference scenarios.


### Pre-tuning Model Shapes for Inference Scenarios

`flag_gems` integrates with [libtune](https://github.com/FlagOpen/FlagGems/tree/master/src/flag_gems/libtune), a lightweight tuning cache mechanism that helps mitigate Triton’s runtime autotuning overhead. Instead of tuning operators on the fly—potentially during the first few inference requests—`libtune` allows you to **pre-tune critical operators for known shapes** and persist the best configurations.

Key characteristics:

- Caches best autotune configs in a **per-device, cross-process database**.
- Eliminates redundant tuning during runtime, which is especially beneficial for large models and high-throughput systems.
- Especially useful for operators like `mm` and `addmm`, which are often heavily autotuned.

To use pre-tuning:

1. Identify the key shapes used in your inference scenario.
2. Use the [`examples/pretune.py`](https://github.com/FlagOpen/FlagGems/blob/master/examples/pretune.py) script to warm up and store the best configs ahead of time.
3. Run your inference workloads with autotuning disabled or minimal, relying on the cached best configs.

> Note: In some frameworks like vLLM (e.g., `v0.8.5`), a built-in model warmup step is available when using `--compile-mode`. If `flag_gems` is integrated, this warmup implicitly triggers pre-tuning via `libtune`.


### Using C++-Based Operator Wrappers for Further Performance Gains

Another advanced optimization path in `flag_gems` is the use of **C++ wrappers** for selected operators. While Triton kernels offer reasonably good compute performance, Triton itself is a Python-embedded DSL. This means that both operator definition and runtime dispatch rely on Python, which can introduce **non-trivial overhead** in latency-sensitive or high-throughput scenarios.

To address this, we provide a C++ runtime solution that encapsulates the operator’s wrapper logic, registration mechanism, and runtime management entirely in C++, while still reusing the underlying Triton kernels for the actual computation. This approach maintains Triton's kernel-level efficiency while significantly reducing Python-related overhead, enabling tighter integration with low-level CUDA workflows and improving overall inference performance.

#### Installation & Setup
To use the C++ operator wrappers:

1. **Follow the [C++ Extension Build Guide](https://github.com/FlagOpen/FlagGems/blob/master/docs/build_flaggems_with_c_extensions.md)** to compile and install the C++ version of `flag_gems`.

2. **Verify successful installation** with the following snippet:
    ```
    try:
        from flag_gems import ext_ops  # noqa: F401
        has_c_extension = True
    except ImportError:
        has_c_extension = False
    ```
	If `has_c_extension` is `True`, then the C++ runtime path is available.

3. When installed successfully, C++ wrappers will automatically be preferred **in patch mode** and when explicitly building models using `flag_gems`-defined modules.  For example, `gems_rms_forward` will by default use the C++ wrapper version of `rms_norm`. You can refer to the actual usage in [normalization.py](https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/modules/normalization.py#L46) to better understand how C++ operator wrappers are integrated and invoked.

#### Explicitly Using C++ Operators

If you want to **directly call C++-wrapped operators**, bypassing any patch logic or fallback, use the `torch.ops.flag_gems` namespace like this:
```
output = torch.ops.flag_gems.fused_add_rms_norm(...)
```
This gives you **precise control** over operator dispatch, which can be beneficial in performance-sensitive contexts.

#### Currently Supported C++-Wrapped Operators

| Operator Name         | Description                        |
|-----------------------|------------------------------------|
| `add`                 | Element-wise addition              |
| `sum`                 | Reduction across dimensions        |
| `rms_norm`            | Root Mean Square normalization     |
| `fused_add_rms_norm`  | Fused addition + RMSNorm           |
| `mm`                  | Matrix multiplication              |

We are actively expanding this list as part of our ongoing performance roadmap.
