import pytest
import torch

import flag_gems

from .accuracy_utils import (
    CAT_SHAPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    gems_assert_equal,
    to_reference,
)


def gen_cat_shapes_dim(shapes):
    results = []
    for tensor_shapes in shapes:
        assert all(
            [len(s) == len(tensor_shapes[0]) for s in tensor_shapes]
        ), "All tensor rank must agree."
        assert all(
            [s[-1] == tensor_shapes[0][-1] for s in tensor_shapes]
        ), "All tensor must have same shape except cat dim."
        rank = len(tensor_shapes[0])
        results.append([tensor_shapes, 0])
        for dim in range(1, rank):
            results.append(
                [[(s[dim], *s[1:dim], s[0], *s[dim + 1 :]) for s in tensor_shapes], dim]
            )
    return results


@pytest.mark.parametrize("shapes, dim", gen_cat_shapes_dim(CAT_SHAPES))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_cat(shapes, dim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(shape, dtype=dtype, device="cuda") for shape in shapes]
    else:
        inp = [
            torch.randint(-1000, 1000, shape, dtype=dtype, device="cuda").to(dtype)
            for shape in shapes
        ]
    ref_inp = [to_reference(_, True) for _ in inp]

    ref_out = torch.cat(ref_inp, dim=dim)

    with flag_gems.use_gems():
        res_out = torch.cat(ref_inp, dim=dim)
    gems_assert_equal(res_out, ref_out)
