import pytest
import torch

from . import base, consts

# Standard shapes: (num_blocks, block_size)
BLOCK_DIAG_SHAPES = [
    (4, 64),
    (8, 128),
    (16, 64),
    (4, 256),
    (8, 256),
]


def _torch_block_diag(tensors):
    """Wrapper to call torch.block_diag with a list of tensors."""
    return torch.block_diag(*tensors)


class BlockDiagBenchmark(base.Benchmark):
    """Benchmark for block_diag."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = BLOCK_DIAG_SHAPES[:]
        self.shape_desc = "num_blocks, block_size"

    def get_case_iter(self, dtype):
        for ordinal, (num_blocks, block_size) in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"block": [block_size, block_size]},
                    params={"num_blocks": num_blocks},
                    builder_args=(num_blocks, block_size),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        num_blocks, block_size = plan.builder_args
        blocks = [
            torch.randn((block_size, block_size), dtype=case.dtype, device=self.device)
            for _ in range(num_blocks)
        ]
        return (blocks,)


@pytest.mark.block_diag
def test_block_diag():
    bench = BlockDiagBenchmark(
        op_name="block_diag",
        torch_op=_torch_block_diag,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
