# How To Use FlagGems
## Basic Usage
To use the `FlagGems` operator library, import it and enable acceleration before running computations. You can enable it globally or temporarily.

### Option 1: Global Enablement

To apply `FlagGems` optimizations across your entire script or interactive session:

```python
import flag_gems

# Enable flag_gems globally
flag_gems.enable()
```
Once enabled, all supported operators in your code will automatically be replaced with the optimized `FlagGems` implementations—no further changes needed.

### Option 2: Scoped Enablement

For finer control, you can enable `FlagGems` only within a specific code block using its context manager:

```python
import flag_gems

# Enable flag_gems temporarily
with flag_gems.use_gems():
    # Code inside this block will use Gems-accelerated operators
    ...

```
This scoped usage is helpful when you want to:

- Benchmark performance differences

- Compare correctness between implementations

- Apply acceleration selectively in complex workflows

## Advanced Usage
The `flag_gems.enable(...)` function supports several optional parameters to give you fine-grained control over how acceleration is applied. This allows for more flexible integration and easier debugging or profiling in complex workflows.
### Parameter Overview

| Parameter      | Type       | Description                                                              |
|----------------|------------|--------------------------------------------------------------------------|
| `unused`       | List[str]  | Disable specific operators                                               |
| `record`       | bool       | Log operator calls for debugging or profiling                            |
| `path`         | str        | Log file path (only used when `record=True`)                             |
| `forward_only` | bool       | Enable acceleration for forward pass only (recommended for inference)    |


### Example : Selectively Disable Specific Operators

You can use the `unused` parameter to exclude certain operators from being accelerated by `FlagGems`. This is especially useful when a particular operator does not behave as expected in your workload, or if you're seeing suboptimal performance and want to temporarily fall back to the original implementation.

```python
flag_gems.enable(unused=["sum", "add"])
```
With this configuration, `sum` and `add` will continue to use the native PyTorch implementations, while all other supported operators will use `FlagGems` versions.

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

### Supported Platforms
| Vendor        | Platform Examples                  | Backend Notes                  |
|---------------|------------------------------------|--------------------------------|
| NVIDIA        | A100, H100                         | Default Triton backend         |
| Others        | -                                  | Coming soon                    |


### Unified Usage Interface

Regardless of the underlying hardware, the usage of `flag_gems` remains exactly the same. There is no need to modify application code when switching from NVIDIA to non-NVIDIA platforms.

Once you call `import flag_gems` and enable acceleration via `flag_gems.enable()`, operator dispatch will automatically route to the correct backend. This provides a consistent developer experience across heterogeneous environments.

### Backend Requirements

Although the usage pattern is unchanged, running on non-NVIDIA hardware requires that the underlying dependencies—**PyTorch** and the **Triton compiler**—are available and properly configured for the target platform.

There are two common ways to obtain compatible builds:

1. **Request from Hardware Vendor**
   Hardware vendors typically maintain custom builds of PyTorch and Triton tailored to their chips. Contact the vendor to request the appropriate versions.

2. **Explore the FlagTree Project**
     The [FlagTree project](https://github.com/FlagTree/flagtree) offers a unified Triton compiler infrastructure that supports a range of AI chips, including NVIDIA and non-NVIDIA platforms. It consolidates vendor-specific patches and enhancements into a shared open-source backend, simplifying compiler maintenance and enabling multi-platform compatibility.

   > ⚠️ FlagTree provides Triton only. A matching PyTorch build is still required separately.

> **Note**: Some platforms may require additional setup or patching.

### Backend Auto-Detection and Manual Setting
By default, `flag_gems` automatically detects the current hardware backend at runtime and selects the corresponding implementation. In most cases, no manual configuration is required, and everything works out of the box.

However, if auto-detection fails or is incompatible with your environment, you can manually set the target backend to ensure correct runtime behavior. To do this, set the following environment variable before running your code:
```
export GEMS_VENDOR=<your_vendor_name>
```
> ⚠️ This setting should match the actual hardware platform. Manually setting an incorrect backend may result in runtime errors.

You can verify the active backend at runtime using:
```
import flag_gems
print(flag_gems.vendor_name)
```

## Integration with Popular Frameworks

To help integrate `flag_gems` into real-world scenarios, we provide examples with widely-used deep learning frameworks. These integrations require minimal code changes and preserve the original workflow structure.

For full examples, see the [`examples/`](https://github.com/FlagOpen/FlagGems/tree/master/examples) directory.

### Example 1: Hugging Face Transformers

Integration with Hugging Face's `transformers` library is straightforward — simply follow the basic usage patterns introduced in previous sections.

During inference, you can activate acceleration without modifying the model or tokenizer logic. Here's a minimal example:
```
from transformers import AutoModelForCausalLM, AutoTokenizer
import flag_gems

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sharpbai/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("sharpbai/Llama-2-7b-hf")

# Move model to correct device and set to eval mode
device = flag_gems.device
model.to(device).eval()

# Prepare input and run inference with flag_gems enabled
inputs = tokenizer(prompt, return_tensors="pt").to(device=device)
with flag_gems.use_gems():
    output = model.generate(**inputs, max_length=100, num_beams=5)
```
This pattern ensures that all compatible operators used during generation will be automatically accelerated.
You can find more examples in the following files:
- `examples/model_llama_test.py`
- `examples/model_llava_test.py`

### Example 2: vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput inference engine designed for serving large language models efficiently. It supports features like paged attention, continuous batching, and optimized memory management.

`flag_gems` can be integrated into vLLM to replace both standard PyTorch (`aten`) ops and vLLM's internal custom kernels.

#### Replacing Standard PyTorch Operators in vLLM
To accelerate standard PyTorch ops (e.g., `add`, `masked_fill`) in vLLM, simply use the same approach as in other frameworks:
- Call `flag_gems.enable()` before any model initialization or inference.
- This overrides all compatible PyTorch `aten` ops, including those indirectly used in vLLM.

#### Replacing vLLM-Specific Custom Operators
To further optimize vLLM’s internal kernels, `flag_gems` provides an additional API:
```
flag_gems.apply_gems_patches_to_vllm(verbose=True)
```
This function patches certain vLLM-specific C++ or Triton operators with `flag_gems` implementations. When `verbose=True`, it will log which functions were replaced:
``` shell
Patched RMSNorm.forward_cuda with FLAGGEMS custom_gems_rms_forward_cuda
Patched RotaryEmbedding.forward_cuda with FLAGGEMS custom_gems_rope_forward_cuda
Patched SiluAndMul.forward_cuda with FLAGGEMS custom_gems_silu_and_mul
```
Use this when more comprehensive `flag_gems` coverage is desired.

#### Full Example: Enable `flag_gems` in vLLM Inference
```
from vllm import LLM, SamplingParams
import flag_gems

# Step 1: Enable acceleration for PyTorch (aten) operators
flag_gems.enable()

# Step 2: (Optional) Patch vLLM custom ops
flag_gems.apply_gems_patches_to_vllm(verbose=True)

# Step 3: Use vLLM as usual
llm = LLM(model="sharpbai/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.8, max_tokens=128)

output = llm.generate("Tell me a joke.", sampling_params)
print(output)
```

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
