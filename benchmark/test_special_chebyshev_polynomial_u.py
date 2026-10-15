import pytest
import torch

from . import base


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"n": 3},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    # torch.special.chebyshev_polynomial_u only supports float32 on CPU/CUDA
    x = torch.randn(shape, dtype=torch.float32, device=device)
    # Use a fixed n value for benchmarking
    n = 3
    return x, n


# torch.special.chebyshev_polynomial_u only supports float32
@pytest.mark.special_chebyshev_polynomial_u
def test_special_chebyshev_polynomial_u():
    bench = base.GenericBenchmarkExcluse1D(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="special_chebyshev_polynomial_u",
        # torch.special.chebyshev_polynomial_u only supports float32 on CPU/CUDA
        dtypes=[torch.float32],
        torch_op=torch.special.chebyshev_polynomial_u,
    )
    bench.run()
