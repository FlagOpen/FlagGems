from typing import Generator

import pytest
import torch

from . import base, consts, utils


class BroadcastTensorsBenchmark(base.Benchmark):
    # broadcast_tensors accepts multiple tensors of different shapes,
    # so shapes here represent the target broadcast output shape.
    DEFAULT_SHAPE_DESC = "target shape"

    def set_shapes(self, shape_file_path=None):
        # Representative sizes covering 2D, 3D and a range of element counts
        # for broadcast_tensors performance measurement.
        self.shapes = [
            (64, 64),
            (256, 256),
            (4096, 4096),
            (64, 512, 512),
            (1024, 1024, 1024),
        ]

    def get_case_iter(self, dtype) -> Generator:
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"target": list(shape)},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        # Produce two tensors with different shapes that require actual broadcasting
        if len(shape) >= 2:
            shape_a = list(shape)
            shape_a[0] = 1
            shape_b = list(shape)
            shape_b[1] = 1
            inp1 = utils.generate_tensor_input(tuple(shape_a), case.dtype, self.device)
            inp2 = utils.generate_tensor_input(tuple(shape_b), case.dtype, self.device)
        else:
            inp1 = utils.generate_tensor_input(shape, case.dtype, self.device)
            inp2 = utils.generate_tensor_input((1,), case.dtype, self.device)
        return inp1, inp2


@pytest.mark.broadcast_tensors
def test_broadcast_tensors():
    bench = BroadcastTensorsBenchmark(
        op_name="broadcast_tensors",
        torch_op=torch.broadcast_tensors,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
