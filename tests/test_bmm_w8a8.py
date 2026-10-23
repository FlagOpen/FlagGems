import pytest
import torch

import flag_gems


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 17, 37, 65),
        (3, 64, 80, 128),
        (2, 11, 13, 0),
        (1, 0, 13, 19),
        (1, 31, 47, 257),
    ],
)
@pytest.mark.parametrize("transpose", [False, True])
def test_bmm_w8a8(dtype, out_dtype, shape, transpose):
    torch.manual_seed(20261019)
    batch, m, n, k = shape
    device = flag_gems.device
    b_shape = (batch, n, k) if transpose else (batch, k, n)
    if dtype == torch.int8:
        a = torch.randint(-128, 128, (batch, m, k), device=device, dtype=dtype)
        b = torch.randint(-128, 128, b_shape, device=device, dtype=dtype)
    else:
        a = (torch.randn(batch, m, k, device=device) * 3).to(dtype)
        b = (torch.randn(b_shape, device=device) * 3).to(dtype)
    if transpose:
        b = b.transpose(1, 2)
    sa = (torch.rand(batch, m * 2, device=device) * 0.005)[:, ::2]
    sb = (torch.rand(batch, n * 2, device=device) * 0.005)[:, ::2]
    out = torch.empty(batch, n, m, device=device, dtype=out_dtype).transpose(1, 2)
    expected = (torch.bmm(a.double(), b.double()) * sa[:, :, None] * sb[:, None, :]).to(
        out_dtype
    )
    actual = flag_gems.bmm_w8a8(
        a, b, sa[:, :, None], sb[:, None, :], out_dtype=out_dtype, out=out
    )
    assert actual is out
    torch.testing.assert_close(
        actual, expected, atol=1e-5, rtol=0.002 if out_dtype == torch.float16 else 0.01
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 97, 80, 128),
        (2, 64, 80, 128),
        (3, 129, 256, 256),
        (1, 257, 128, 7168),
        (2, 64, 32, 32),
        (1, 163, 2112, 1024),
    ],
)
def test_bmm_w8a8_aligned(shape):
    torch.manual_seed(20261020)
    batch, m, n, k = shape
    a = torch.randint(
        -128, 128, (batch, m, k), device=flag_gems.device, dtype=torch.int8
    )
    b = torch.randint(
        -128, 128, (batch, n, k), device=flag_gems.device, dtype=torch.int8
    ).transpose(1, 2)
    sa = torch.rand(batch, m, device=flag_gems.device) * 0.001
    sb = torch.rand(batch, n, device=flag_gems.device) * 0.001
    expected = (
        torch.bmm(a.double(), b.double()) * sa[:, :, None] * sb[:, None, :]
    ).bfloat16()
    actual = flag_gems.bmm_w8a8(a, b, sa, sb)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.01)


@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX graph fast path")
def test_bmm_w8a8_graph_replay():
    torch.manual_seed(20261020)
    a = torch.randint(
        -128, 128, (2, 64, 128), device=flag_gems.device, dtype=torch.int8
    )
    b = torch.randint(
        -128, 128, (2, 80, 128), device=flag_gems.device, dtype=torch.int8
    ).transpose(1, 2)
    sa = torch.rand(2, 64, device=flag_gems.device) * 0.001
    sb = torch.rand(2, 80, device=flag_gems.device) * 0.001
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flag_gems.bmm_w8a8(a, b, sa, sb)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = flag_gems.bmm_w8a8(a, b, sa, sb)
    for _ in range(3):
        a.random_(-128, 128)
        sa.uniform_(0, 0.001)
        sb.uniform_(0, 0.001)
        graph.replay()
        torch.cuda.synchronize()
        expected = (
            torch.bmm(a.double(), b.double()) * sa[:, :, None] * sb[:, None, :]
        ).bfloat16()
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=0.01)
