import pytest
import torch

from . import base, consts


class UpsampleNearest3dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Typical volumetric feature map sizes: small, medium, large spatial dims
        self.shapes = [(2, 3, 8, 8, 8), (4, 8, 16, 16, 16), (4, 16, 24, 24, 24)]

    def set_more_shapes(self):
        return None

    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            out_d = shape[2] * 2
            out_h = shape[3] * 2
            out_w = shape[4] * 2
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": [shape[0], shape[1], out_d, out_h, out_w]},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        x = torch.randn(shape, dtype=case.dtype, device=self.device)
        out_d = shape[2] * 2
        out_h = shape[3] * 2
        out_w = shape[4] * 2
        output_size = (out_d, out_h, out_w)

        out = torch.ops.aten.upsample_nearest3d(
            x, [out_d, out_h, out_w], None, None, None
        )
        grad_output = torch.ones_like(out)

        input_size = tuple(x.shape)
        return grad_output, output_size, input_size


class UpsampleNearest3dBackwardGradInputBenchmark(UpsampleNearest3dBackwardBenchmark):
    # The .grad_input overload requires the out tensor `grad_input` to be
    # supplied. Yield it as a kwargs dict so it is forwarded as a keyword
    # argument (see base.unpack_to_args_kwargs).
    def build_inputs(self, case):
        grad_output, output_size, input_size = super().build_inputs(case)
        grad_input = torch.empty(input_size, dtype=case.dtype, device=self.device)
        return grad_output, output_size, input_size, {"grad_input": grad_input}


@pytest.mark.upsample_nearest3d_backward
def test_upsample_nearest3d_backward():
    bench = UpsampleNearest3dBackwardBenchmark(
        op_name="upsample_nearest3d_backward",
        torch_op=torch.ops.aten.upsample_nearest3d_backward.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.upsample_nearest3d_backward_grad_input
def test_upsample_nearest3d_backward_grad_input():
    bench = UpsampleNearest3dBackwardGradInputBenchmark(
        op_name="upsample_nearest3d_backward_grad_input",
        torch_op=torch.ops.aten.upsample_nearest3d_backward.grad_input,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
