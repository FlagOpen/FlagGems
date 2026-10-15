import pytest
import torch

from . import base, consts

# Custom shapes for _chunk_cat benchmark: 1D and 2D tensors
_CHUNK_CAT_SHAPES = [
    (16,),
    (32,),
    (64,),
    (128,),
    (256,),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
]


class ChunkCatBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _CHUNK_CAT_SHAPES

    def get_case_iter(self, cur_dtype):
        ordinal = 0
        for shape in self.shapes:
            for num_chunks in [2, 4]:
                yield self._case_from_plan(
                    cur_dtype,
                    ordinal,
                    base.BenchmarkCasePlan(
                        shape={"input": list(shape)},
                        params={"dim": 0, "num_chunks": num_chunks},
                        builder_args=(shape, num_chunks),
                    ),
                )
                ordinal += 1

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, num_chunks = plan.builder_args
        inp = torch.randn(shape, dtype=case.dtype, device=self.device)
        return [inp], 0, num_chunks


@pytest.mark.chunk_cat
def test_chunk_cat():
    bench = ChunkCatBenchmark(
        op_name="chunk_cat",
        torch_op=torch._chunk_cat,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
