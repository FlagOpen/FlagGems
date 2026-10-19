import pytest
import torch

from . import base


HSPMM_SHAPES = [
    (256, 64, 32, 64),
    (5000, 512, 64, 512),
    (20000, 1024, 128, 1024),
    (100000, 4096, 256, 1024),
]


def _make_sparse_coo(shape, nnz, dtype, device):
    M, K = shape
    indices = torch.stack(
        [torch.randint(0, M, (nnz,), device=device), torch.randint(0, K, (nnz,), device=device)]
    )
    values = torch.randn(nnz, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=dtype, device=device).coalesce()


class HspmmBenchmark(base.GenericBenchmark):
    DEFAULT_METRICS = base.BlasBenchmark.DEFAULT_METRICS

    def set_shapes(self, shape_file_path=None):
        self.shapes = HSPMM_SHAPES

    def get_tflops(self, op, *args, **kwargs):
        mat1, mat2 = args
        return mat1._nnz() * mat2.shape[-1] * 2


def _case_fn(shape, dtype):
    del dtype
    nnz, m, n, k = shape
    yield base.BenchmarkCasePlan(
        shape={"mat1": [m, k], "mat2": [k, n]},
        params={"nnz": nnz},
        builder_args=(nnz, m, n, k),
    )


def _build_inputs_fn(plan, dtype, device):
    nnz, m, n, k = plan.builder_args
    return _make_sparse_coo((m, k), nnz, dtype, device), torch.randn((k, n), dtype=dtype, device=device)


@pytest.mark.hspmm
def test_hspmm():
    HspmmBenchmark(
        op_name="hspmm",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.hspmm,
        dtypes=[torch.float32, torch.float64],
    ).run()
