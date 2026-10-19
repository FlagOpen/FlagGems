# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import gc
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

import torch

from flag_gems import gdn_w8a16

ROOT = Path(os.environ.get("GDN_RESULT_DIR", "gdn_w8a16_results"))
SHAPES = Path(
    os.environ.get(
        "GDN_SHAPE_FILE", str(Path(__file__).parent / "shapes" / "gdn_w8a16.json")
    )
)


def make_inputs(row):
    shape = row["shape"]
    if len(shape) == 4:
        tokens, _, hv, dv = shape
        heads, dk, n = 16, 128, tokens
        packed = torch.randn(
            tokens, 2 * heads * dk + hv * dv, device="cuda", dtype=torch.bfloat16
        )
        q = packed[:, : heads * dk].view(1, tokens, heads, dk)
        k = packed[:, heads * dk : 2 * heads * dk].view(1, tokens, heads, dk)
        v = packed[:, 2 * heads * dk :].view(1, tokens, hv, dv)
    else:
        _, tokens, heads, dk, hv, dv, n = shape
        q = torch.randn(1, tokens, heads, dk, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, tokens, hv, dv, device="cuda", dtype=torch.bfloat16)
    # Raw normal draws; no per-vector normalization. Small beta keeps the
    # recurrence stable with unit-variance keys (expected squared norm = K).
    g = -torch.rand(1, tokens, hv, device="cuda") * 0.1
    beta = torch.rand(1, tokens, hv, device="cuda") / dk
    state = torch.randn(n, hv, dv, dk, device="cuda") * 0.1
    bounds = [i * tokens // n for i in range(n + 1)]
    cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    return (q, k, v, g, beta, state, cu), bounds


def torch_baseline(args, bounds):
    q, k, v, g, beta, state, _ = args
    if len(bounds) - 1 == q.shape[1]:
        group = v.shape[2] // q.shape[2]
        qq = q[0].float().repeat_interleave(group, 1)
        kk = k[0].float().repeat_interleave(group, 1)
        vv = v[0].float()
        s = state * g[0].exp()[..., None, None]
        delta = (vv - torch.matmul(s, kk.unsqueeze(-1)).squeeze(-1)) * beta[
            0, :, :, None
        ]
        s = s + delta.unsqueeze(-1) * kk.unsqueeze(-2)
        o = torch.matmul(s, qq.unsqueeze(-1)).squeeze(-1) * (q.shape[-1] ** -0.5)
        return o.unsqueeze(0).to(q.dtype), s
    return torch_prefill(args, bounds)


def torch_prefill(args, bounds):
    q, k, v, g, beta, state, _ = args
    group = v.shape[2] // q.shape[2]
    q = q[0].float().repeat_interleave(group, 1).transpose(0, 1)
    k = k[0].float().repeat_interleave(group, 1).transpose(0, 1)
    v = v[0].float().transpose(0, 1)
    g = g[0].transpose(0, 1)
    beta = beta[0].transpose(0, 1)
    outputs, states = [], []
    for seq, (start, stop) in enumerate(zip(bounds, bounds[1:])):
        length = stop - start
        chunks = (length + 63) // 64
        padding = chunks * 64 - length
        qq = torch.nn.functional.pad(q[:, start:stop], (0, 0, 0, padding)).reshape(
            q.shape[0], chunks, 64, q.shape[-1]
        )
        kk = torch.nn.functional.pad(k[:, start:stop], (0, 0, 0, padding)).reshape(
            k.shape[0], chunks, 64, k.shape[-1]
        )
        vv = torch.nn.functional.pad(v[:, start:stop], (0, 0, 0, padding)).reshape(
            v.shape[0], chunks, 64, v.shape[-1]
        )
        gg = (
            torch.nn.functional.pad(g[:, start:stop], (0, padding))
            .reshape(g.shape[0], chunks, 64)
            .cumsum(-1)
        )
        bb = torch.nn.functional.pad(beta[:, start:stop], (0, padding)).reshape(
            beta.shape[0], chunks, 64
        )
        decay = (gg[..., :, None] - gg[..., None, :]).exp().tril()
        lhs = (kk @ kk.transpose(-1, -2) * decay * bb[..., None]).tril(-1)
        # Strictly lower 64x64 L is nilpotent: the finite geometric series
        # gives (I+L)^-1 exactly, using only PyTorch batched matmuls.
        power = -lhs
        inverse = torch.eye(64, device=q.device) + power
        for _ in range(5):
            power = power @ power
            inverse = inverse + inverse @ power
        rhs = bb[..., None] * torch.cat((vv, gg.exp()[..., None] * kk), dim=-1)
        solved = inverse @ rhs
        u, w = solved.split((vv.shape[-1], kk.shape[-1]), dim=-1)
        q_exp = qq * gg.exp()[..., None]
        qk = (qq @ kk.transpose(-1, -2)) * decay
        weighted_k = kk * (gg[..., -1:] - gg).exp()[..., None]
        last_decay = gg[..., -1].exp()
        s = state[seq].transpose(-1, -2)
        chunk_outputs = []
        for i in range(chunks):
            update = u[:, i] - w[:, i] @ s
            chunk_outputs.append(q_exp[:, i] @ s + qk[:, i] @ update)
            s = (
                last_decay[:, i, None, None] * s
                + weighted_k[:, i].transpose(-1, -2) @ update
            )
        outputs.append(
            torch.stack(chunk_outputs, dim=1).reshape(
                q.shape[0], chunks * 64, v.shape[-1]
            )[:, :length]
        )
        states.append(s.transpose(-1, -2))
    output = torch.cat(outputs, dim=1).transpose(0, 1).unsqueeze(0) * (
        q.shape[-1] ** -0.5
    )
    return output.to(args[0].dtype), torch.stack(states).contiguous()


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    return graph, result


def measure_pair(first, second, samples=31):
    values = [[], []]
    for i in range(samples):
        for which in ((0, 1) if i % 2 == 0 else (1, 0)):
            graph = (first, second)[which]
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            begin.record()
            for _ in range(3):
                graph.replay()
            end.record()
            end.synchronize()
            values[which].append(begin.elapsed_time(end) * 1000 / 3)
    return values


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261019)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = json.loads(SHAPES.read_text())
    limit = int(os.environ.get("GDN_LIMIT", "0"))
    if limit:
        rows = rows[:limit]
    results = []
    metadata = dict(
        operator="gdn_w8a16",
        implementation="basic Triton recurrence",
        baseline=(
            "PyTorch batched recurrence for single-token sequences; "
            "FP32 batched chunk, finite-series triangular inverse via matmul, "
            "plus state scan for prefill"
        ),
        timing="CUDA Graph, 31 alternating-order samples, each 3 replays, median us per replay",
        dtype="BF16 q/k/v, FP32 g/beta/state and accumulation; BF16 output",
        input_distribution="raw q/k/v N(0,1), g=-U(0,0.1), beta=U(0,1)/K, state=N(0,0.1); no q/k normalization",
        scope="GDN recurrence core, precomputed g/beta; no projection or sigmoid/softplus",
        sequence_boundaries="floor(i*T/N), exact original per-request lengths unavailable",
        torch=torch.__version__,
        seed=20261019,
        gpu=torch.cuda.get_device_name(0),
    )
    for i, row in enumerate(rows):
        start = time.monotonic()
        args, bounds = make_inputs(row)
        fn = partial(gdn_w8a16, *args)
        baseline = partial(torch_baseline, args, bounds)
        print("START", i + 1, row["shape"], flush=True)
        actual = fn()
        expected = baseline()
        for a, e in zip(actual, expected):
            if not torch.isfinite(a).all() or not torch.isfinite(e).all():
                raise RuntimeError("Non-finite output/state")
            torch.testing.assert_close(a.float(), e.float(), atol=0.002, rtol=0.02)
        print("CAPTURE", i + 1, flush=True)
        graph, out = capture(fn)
        base_graph, base_out = capture(baseline)
        for a, e in zip(out, base_out):
            torch.testing.assert_close(a.float(), e.float(), atol=0.002, rtol=0.02)
        candidate, base = measure_pair(graph, base_graph)
        us, base_us = statistics.median(candidate), statistics.median(base)
        results.append(
            dict(
                row,
                core_us=us,
                baseline_us=base_us,
                speedup=base_us / us,
                status="pass",
                output_max_abs=float(
                    (actual[0].float() - expected[0].float()).abs().max()
                ),
                state_max_abs=float((actual[1] - expected[1]).abs().max()),
                samples_us=candidate,
                baseline_samples_us=base,
            )
        )
        (ROOT / "results.json").write_text(
            json.dumps(dict(metadata=metadata, results=results), indent=2)
        )
        print(
            i + 1,
            row["op"],
            row["shape"],
            "candidate_us",
            round(us, 3),
            "torch_us",
            round(base_us, 3),
            "speedup",
            round(base_us / us, 4),
            "sec",
            round(time.monotonic() - start, 2),
            flush=True,
        )
        del graph, base_graph, out, base_out, actual, expected, args, fn, baseline
        gc.collect()
        torch.cuda.empty_cache()
    print("ALL PASSED", flush=True)


if __name__ == "__main__":
    main()
