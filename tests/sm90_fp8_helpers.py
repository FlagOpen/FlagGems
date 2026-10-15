# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent numerical references and deterministic synthetic fixtures."""

import torch


def fp8_inputs(m, n, k, seed=41):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(m, k, device="cuda", generator=g) * 0.25).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, device="cuda", generator=g) * 0.25).to(torch.float8_e4m3fn)
    sa = torch.exp2(
        torch.randint(-3, 3, (m, k // 32), device="cuda", generator=g).float()
    )
    sb = torch.exp2(
        torch.randint(
            -3, 3, ((n + 31) // 32, k // 32), device="cuda", generator=g
        ).float()
    )
    return a, b, sa, sb


def fp8_reference(a, b, sa, sb, output_dtype=torch.bfloat16, **_):
    m, k = a.shape
    n = b.shape[0]
    out = torch.zeros((m, n), dtype=torch.float32, device=a.device)
    for i in range(k // 32):
        sl = slice(32 * i, 32 * (i + 1))
        dot = a[:, sl].float() @ b[:, sl].float().T
        out += dot * sa[:, i, None] * sb[:, i].repeat_interleave(32)[:n][None, :]
    return out.to(output_dtype)


def assert_exact(a, b):
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_exact(x, y)
    else:
        assert a.shape == b.shape and a.dtype == b.dtype
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        ), "visible storage bytes differ"


def changed_graph(fn, args, refresh, reference, repeats=3):
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(*args)
    for seed in range(repeats):
        refresh(args, seed)
        graph.replay()
        torch.cuda.synchronize()
        assert_exact(out, reference(*args))
