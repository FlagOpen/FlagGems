import pytest
import torch

from . import consts
from .test_upsample_nearest_exact2d import UpsampleNearestExact2dBenchmark


@pytest.mark.upsample_nearest_exact2d_out
def test_upsample_nearest_exact2d_out():
    UpsampleNearestExact2dBenchmark(
        op_name="upsample_nearest_exact2d_out",
        torch_op=torch.ops.aten._upsample_nearest_exact2d.out,
        dtypes=consts.FLOAT_DTYPES,
        with_out=True,
        is_inplace=True,
    ).run()
