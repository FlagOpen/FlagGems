import gc
import json
import os
import statistics

import torch

from flag_gems import bmm_w8a8


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        o = fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    return g, o


def measure(gs):
    values = [[] for g in gs]
    for i in range(31):
        for j in (range(len(gs)) if i % 2 else reversed(range(len(gs)))):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            a.record()
            for _ in range(5):
                gs[j].replay()
            b.record()
            b.synchronize()
            values[j].append(a.elapsed_time(b) * 200)
    return values


def main():
    torch.manual_seed(20261019)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    dtypes = (
        (torch.int8,)
        if os.environ.get("BMM_INT8_ONLY") == "1"
        else (torch.int8, torch.float8_e4m3fn, torch.float8_e5m2)
    )
    for dt in dtypes:
        for m, n, k in [
            (16384, 7168, 1024),
            (16384, 2112, 7168),
            (448, 7168, 256),
            (14429, 7168, 1024),
            (14429, 2112, 7168),
        ]:
            bs = 1
            raw_a = torch.randn(bs, m, k, device="cuda", dtype=torch.bfloat16)
            raw_b = torch.randn(
                bs, n, k, device="cuda", dtype=torch.bfloat16
            ).transpose(1, 2)
            limit = (
                127
                if dt == torch.int8
                else (448 if dt == torch.float8_e4m3fn else 57344)
            )
            sa = raw_a.float().abs().amax(2) / limit
            sb = raw_b.float().abs().amax(1) / limit
            aq = raw_a.float() / sa[:, :, None]
            bq = raw_b.float() / sb[:, None, :]
            if dt == torch.int8:
                aq = aq.round()
                bq = bq.round()
            a = aq.to(dt)
            b = bq.transpose(1, 2).contiguous().to(dt).transpose(1, 2)
            ref = (
                torch.bmm(a.float(), b.float()) * sa[:, :, None] * sb[:, None, :]
            ).bfloat16()
            f = lambda a=a, b=b, sa=sa, sb=sb: bmm_w8a8(a, b, sa, sb)
            base = lambda a=a, b=b, sa=sa, sb=sb: (
                torch.bmm(a.float(), b.float()) * sa[:, :, None] * sb[:, None, :]
            ).bfloat16()
            # Raw BF16 BMM is a separate lower-work baseline, not numerically equivalent.
            ab = raw_a
            bb = raw_b
            raw = lambda ab=ab, bb=bb: torch.bmm(ab, bb)
            out = f()
            torch.testing.assert_close(out, ref, atol=0.002, rtol=0.01)
            gs = []
            outs = []
            for fn in (f, base, raw):
                g, o = capture(fn)
                gs.append(g)
                outs.append(o)
            samples = measure(gs)
            us = [statistics.median(x) for x in samples]
            rows.append(
                dict(
                    dtype=str(dt),
                    shape=[bs, m, n, k],
                    batch_assumed=True,
                    candidate_us=us[0],
                    torch_channel_us=us[1],
                    raw_bf16_us=us[2],
                    speedup=us[1] / us[0],
                    raw_speedup=us[2] / us[0],
                    samples=samples,
                    correct=True,
                )
            )
            json.dump(
                rows, open(os.environ.get("BMM_RESULT", "results.json"), "w"), indent=2
            )
            print(
                rows[-1]["dtype"],
                [bs, m, n, k],
                us,
                "speedup",
                us[1] / us[0],
                "raw",
                us[2] / us[0],
                flush=True,
            )
            del gs, outs, g, o, out, ref, a, b, ab, bb, raw_a, raw_b, aq, bq
            gc.collect()
            torch.cuda.empty_cache()
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
