import pytest
import torch

import flag_gems

from . import base


def choose_qparams_optimized_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"n_bins": 200, "ratio": 0.16, "bit_width": 8},
        builder_args=(shape,),
    )


def choose_qparams_optimized_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    numel = 1
    for s in shape:
        numel *= s
    inp = torch.randn(shape, dtype=dtype, device=device).view(-1)
    # (input, numel, n_bins, ratio, bit_width)
    return inp, numel, 200, 0.16, 8


def torch_choose_qparams_optimized(inp, numel, n_bins, ratio, bit_width):
    # The aten reference only has a CPU implementation, so run it on CPU.
    return torch.choose_qparams_optimized(inp.cpu(), numel, n_bins, ratio, bit_width)


class ChooseQParamsBenchmark(base.GenericBenchmark):
    # The aten reference is an iterative CPU-only kernel, so keep the element
    # counts modest to keep the baseline tractable. Override set_shapes so that
    # core_shapes.yaml can't inject huge shapes that would make the CPU-only
    # reference intractably slow.
    def set_shapes(self, shape_file_path=None):
        # Fixed, hand-picked shapes for this CPU-only iterative reference.
        self.shapes = [(256,), (1024,), (4096,)]

    def set_more_shapes(self):
        return []


@pytest.mark.choose_qparams_optimized
def test_choose_qparams_optimized():
    bench = ChooseQParamsBenchmark(
        op_name="choose_qparams_optimized",
        torch_op=torch_choose_qparams_optimized,
        gems_op=flag_gems.choose_qparams_optimized,
        # float32-only: the aten reference has a CPU-only float32 implementation.
        dtypes=[torch.float32],
        case_fn=choose_qparams_optimized_case_fn,
        build_inputs_fn=choose_qparams_optimized_build_inputs_fn,
    )
    bench.run()
