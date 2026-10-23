---
title: "Reference as candidate"
---

`--ref-as-candidate` checks whether a correctness test or benchmark can execute
using its explicitly declared reference as the candidate. It does not require a
candidate source file. The option is off by default; existing tests and override
commands keep their original behavior.

## Correctness contract

Use `@reference_test` on a parametrized factory returning `ReferenceCase`:

```python
import pytest
import torch
import flag_gems
from flag_gems.testing.reference import ReferenceCase, reference_test
from . import accuracy_utils as utils

@pytest.mark.add
@pytest.mark.parametrize("shape", [(16,), (32,)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@reference_test
def test_add(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn_like(x)

    def reference(args):
        return torch.add(*args)

    return ReferenceCase(
        inputs=(x, y),
        candidate=lambda args: flag_gems.add(*args),
        reference=reference,
        compare=lambda actual, expected: utils.gems_assert_close(actual, expected, dtype),
        reference_inputs=(utils.to_reference(x, True), utils.to_reference(y, True)),
    )
```

Both callables receive one input bundle. The default mode executes the real
candidate; the new mode executes `reference` in its place. The existing comparator
still runs. Keep the candidate lookup inside its callable so CLI overrides remain
effective and a missing public implementation need not be resolved in ref mode.

`reference_inputs` optionally supplies the expected-side CPU/upcast inputs; by
default both sides start from `inputs`. The runner clones each bundle separately
before either operation. The default clone is `copy.deepcopy`; supply `clone`
when tensors need custom view/storage/gradient handling. The clone must preserve
within-bundle alias relationships while making the two executions independent.
Return mutated buffers and other observable outputs from each callable, and
retain device, dtype, layout, mutation and alias checks in `compare`. CPU-oracle
references must explicitly implement the candidate's required output contract;
the runner does not silently move outputs or change dtype.

A missing decorator, missing/non-callable field or wrong factory return is an
error with the test node ID. Merely defining a local function named `reference`
does not expose the contract. Unconverted tests still run normally without the
new option. Existing pytest skips remain skips, not evidence of validation.

```sh
python -m pytest tests/test_add.py -m add --ref-as-candidate
python -m pytest tests/test_add.py -m add --ref-as-candidate --ref cpu
```

These commands require the selected tests to implement the contract above. Run
the full parameter grid for acceptance; `--quick` is only a preliminary check.
Correctness JSON records identify `candidate_source: reference` and count both
reference and candidate-side calls.

## Benchmark contract

Standard `Benchmark` subclasses already declare their reference as `torch_op`.
With the new option, that callable is also used for candidate timing, outside
Gems dispatch. Input generation, selected cases, timing mode and metrics remain
in use. The candidate latency metric is required; records identify
`candidate_source: reference`.

```sh
python -m pytest benchmark/test_add.py -m add --level core --ref-as-candidate
```

A non-callable `torch_op` or a custom override of `run`, `_measure_input`,
`get_latency` or `_candidate_call` is an explicit contract error in this mode.
The original behavior of those benchmarks is unchanged without the option.

The option cannot be combined with external overrides, `--reference-only`,
`--preflight-only`, `--profile-only`, `--list-cases`, `--query`, `--parallel`, or
`--collect-marks`. Those modes retain their existing meanings. This mode performs
real timing; `--reference-only` still performs its original baseline-only probe.

Ref-vs-ref is a harness executability check, not independent evidence that a
reference is correct or that external candidate injection works. Continue to
use dummy candidate injection to check the actual override route. These timings
are diagnostic and do not replace the H800 ops-baseline collection workflow.
