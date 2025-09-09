import math
import os
from typing import Any, List, Optional

import pytest
import torch
import triton

import flag_gems

from .performance_utils import Benchmark, GenericBenchmark, SkipVersion, vendor_name


class AttentionBenchmark(GenericBenchmark):
    """
    benchmark for attention
    """

    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (batch, num_heads, seq_len, head_size).
        return None


@pytest.mark.skipif(vendor_name == "metax", reason="TODOFIX")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(vendor_name == "mthreads", reason="RuntimeError")
@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize("dropout_p", [0.0, 0.25])
@pytest.mark.parametrize("is_causal", [True, False])
def test_perf_scaled_dot_product_attention(dropout_p, is_causal):
    def scaled_dot_product_attention_kwargs(shape, dtype, device):
        query = torch.randn(shape, device=device, dtype=dtype)
        key = torch.randn(shape, device=device, dtype=dtype)
        value = torch.randn(shape, device=device, dtype=dtype)
        yield query, key, value, dropout_p, is_causal

    def sdpa_flash(query, key, value, dropout_p=dropout_p, is_causal=is_causal):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    bench = AttentionBenchmark(
        op_name="scaled_dot_product_attention",
        input_fn=scaled_dot_product_attention_kwargs,
        # torch_op=torch.nn.functional.scaled_dot_product_attention,
        torch_op=sdpa_flash,
        dtypes=[
            torch.float16,
            torch.bfloat16,
        ],
    )
    bench.run()


class FlashMLABenchmark(GenericBenchmark):
    """
    benchmark for flash_mla
    """

    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (batch, num_heads, seq_len, head_size).
        return None


@pytest.mark.skipif(vendor_name == "metax", reason="TODOFIX")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(vendor_name == "mthreads", reason="RESULT TODOFIX")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="TypeError")
@pytest.mark.flash_mla
def test_perf_flash_mla():
    def flash_mla_kwargs(shape, dtype, device):
        seqlen = shape[0]
        b = 128
        s_q = 1
        h_q = 128
        h_kv = 1
        d = 576
        dv = 512
        causal = True
        block_size = 64
        cache_seqlens = torch.tensor(
            [seqlen + 2 * i for i in range(b)], dtype=torch.int32, device=device
        )
        max_seqlen = cache_seqlens.max().item()
        max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256

        q = torch.randn([b, s_q, h_q, d], dtype=dtype, device=device)
        block_table = torch.arange(
            b * max_seqlen_pad // block_size, dtype=torch.int32, device=device
        ).view(b, max_seqlen_pad // block_size)
        blocked_k = torch.randn(
            [block_table.numel(), block_size, h_kv, d], dtype=dtype, device=device
        )
        yield q, block_table, blocked_k, max_seqlen_pad, block_size, b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal

    def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
        query = query.float()
        key = key.float()
        value = value.float()
        key = key.repeat_interleave(h_q // h_kv, dim=0)
        value = value.repeat_interleave(h_q // h_kv, dim=0)
        attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        if is_causal:
            s_q = query.shape[-2]
            s_k = key.shape[-2]
            attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device=query.device)
            temp_mask = torch.ones(
                s_q, s_k, dtype=torch.bool, device=query.device
            ).tril(diagonal=s_k - s_q)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)
            attn_weight += attn_bias
        lse = attn_weight.logsumexp(dim=-1)
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        return attn_weight @ value, lse

    def ref_mla(
        q,
        block_table,
        blocked_k,
        max_seqlen_pad,
        block_size,
        b,
        s_q,
        cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        causal,
    ):
        device = q.device
        blocked_v = blocked_k[..., :dv]
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32, device=device)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32, device=device)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    bench = FlashMLABenchmark(
        op_name="flash_mla",
        input_fn=flash_mla_kwargs,
        torch_op=ref_mla,
        dtypes=[
            torch.bfloat16,
        ],
    )
    bench.set_gems(flag_gems.flash_mla)
    bench.run()


class FlashAttnVarlenBenchmark(Benchmark):
    """
    benchmark for flash_attn_varlen_func
    """

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        # Collecting from qwen/Qwen3-1.7B --random-input 512 --random-output 2048 --num-prompts 200 --request-rate inf
        # Format: (seq_lens, num_heads, head_size, block_size, num_blocks, alibi, soft_cap)
        # ([(1, 1), (1, 1), (1, 1)], (16, 8), 128, 32, 18208, False, None),
        # The performance is very poor, which may be related to prefill
        # flash_attn_configs = [
        #     ([(1, 1), (1, 1), (1, 1)], (16, 8), 128, 32, 18208, False, None),
        #     ([(1, 1), (1, 1), (23, 23)], (16, 8), 128, 32, 18208, False, None),
        #     ([(1, 1), (1, 1), (7, 7)], (16, 8), 128, 32, 18208, False, None),
        #     ([(1, 1), (1, 1), (39, 39)], (16, 8), 128, 32, 18208, False, None),
        #     ([(1, 1), (1, 1), (55, 55)], (16, 8), 128, 32, 18208, False, None),
        #     ([(1, 1), (1, 1), (70, 70)], (16, 8), 128, 32, 18208, False, None),
        # ]

        all_cu_seq_lens_q = [
            (0, 512),
            (0, 1, 2, 72),
            (
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                57,
                73,
                89,
                105,
                121,
                137,
                153,
                169,
                185,
                201,
                217,
                233,
                249,
                265,
            ),
            (
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
                116,
                117,
                118,
                119,
                120,
                121,
                122,
                123,
                124,
                125,
                126,
                127,
                128,
                129,
                130,
                131,
                132,
                133,
                134,
                135,
                136,
                137,
                138,
                139,
                140,
                141,
                142,
                143,
                144,
                145,
                146,
                147,
                148,
                149,
                150,
                151,
                152,
                153,
                154,
                155,
                156,
                157,
                158,
                159,
                160,
                161,
                162,
                163,
                164,
                165,
                166,
                167,
                168,
                169,
                170,
                171,
                172,
                173,
                174,
                175,
                176,
                177,
                178,
                179,
                180,
                181,
                182,
                183,
                184,
                185,
                186,
                187,
                188,
                189,
                190,
                191,
                192,
                193,
                194,
                195,
                211,
                226,
                240,
                253,
                265,
            ),
        ]
        all_seqused_k = [
            (512,),
            (
                1,
                1,
                70,
            ),
            (
                515,
                514,
                514,
                514,
                514,
                514,
                514,
                514,
                514,
                514,
                514,
                514,
                514,
                514,
                514,
                514,
                514,
                514,
                514,
                514,
                514,
                513,
                513,
                513,
                513,
                513,
                513,
                513,
                513,
                513,
                513,
                513,
                513,
                513,
                513,
                513,
                513,
                513,
                513,
                513,
                513,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
            ),
            (
                2332,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2331,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2330,
                2329,
                2329,
                2329,
                2329,
                2329,
                2329,
                2329,
                2329,
                2329,
                2329,
                2329,
                2329,
                2329,
                2329,
                2328,
                2328,
                2328,
                2328,
                2328,
                2328,
                2328,
                2328,
                2328,
                2328,
                2328,
                2328,
                2328,
                2328,
                2328,
                2328,
                2328,
                2328,
                2327,
                2327,
                2327,
                2327,
                2327,
                2327,
                2327,
                2327,
                2327,
                2327,
                2327,
                2327,
                2327,
                2327,
                2327,
                2326,
                2326,
                2326,
                2326,
                2326,
                2326,
                2326,
                2326,
                2326,
                2326,
                2326,
                2326,
                2326,
                2326,
                2326,
                2326,
                2326,
                2325,
                2325,
                2325,
                2325,
                2325,
                2325,
                2325,
                2325,
                2325,
                2325,
                2325,
                2325,
                2325,
                2325,
                2325,
                2325,
                2325,
                2325,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2324,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2323,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2322,
                2321,
                2321,
                2321,
                2321,
                2321,
                2320,
                2319,
                2318,
                2317,
                2316,
            ),
        ]

        num_heads = 16
        num_heads_k = 8
        head_dim = 128
        block_size = 16
        num_blocks = 20000
        alibi = False
        soft_cap = None

        all_configs = [
            (
                cu_seq_lens_q,
                seqused_k,
                num_heads,
                num_heads_k,
                head_dim,
                block_size,
                num_blocks,
                alibi,
                soft_cap,
            )
            for cu_seq_lens_q, seqused_k in zip(all_cu_seq_lens_q, all_seqused_k)
        ]

        self.shapes = all_configs

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self.flash_attn_varlen_input_fn(config, cur_dtype, self.device)

    def flash_attn_varlen_input_fn(self, config, dtype, device):
        """Input function for flash attention varlen benchmark"""
        (
            cu_query_lens,
            seqused_k,
            num_query_heads,
            num_kv_heads,
            head_size,
            block_size,
            num_blocks,
            alibi,
            soft_cap,
        ) = config

        if alibi is True and soft_cap is not None:
            return

        num_seqs = len(cu_query_lens) - 1
        max_query_len = max(
            map(lambda x, y: x - y, cu_query_lens[1:], cu_query_lens[:-1])
        )
        max_kv_len = max(seqused_k)
        window_size = (-1, -1)
        scale = head_size**-0.5

        assert num_seqs == len(seqused_k)

        query = torch.randn(
            cu_query_lens[-1], num_query_heads, head_size, dtype=dtype, device=device
        )
        key_cache = torch.randn(
            num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
        )
        value_cache = torch.randn_like(key_cache)
        cu_query_lens = torch.tensor(cu_query_lens, dtype=torch.int32, device=device)
        seqused_k = torch.tensor(seqused_k, dtype=torch.int32, device=device)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

        causal = True

        if alibi:
            alibi_slopes = (
                torch.ones(
                    num_seqs, num_query_heads, device=device, dtype=torch.float32
                )
                * 0.3
            )
        else:
            alibi_slopes = None

        yield (
            query,
            key_cache,
            value_cache,
            max_query_len,
            cu_query_lens,
            max_kv_len,
            None,
            seqused_k,
            None,
            0.0,
            scale,
            causal,
            window_size,
            soft_cap if soft_cap is not None else 0,
            alibi_slopes,
            False,
            False,
            block_tables,
            False,
            None,
            None,
            None,
            None,
            None,
            0,
            2,
        )


@pytest.mark.skipif(
    SkipVersion("vllm", "<0.9"),
    reason="The version prior to 0.9 does not include the flash_attn_varlen_func API in vllm.",
)
@pytest.mark.skipif(
    SkipVersion("torch", "<2.7"),
    reason="The version prior to 2.7 is not compatible with VLLM.",
)
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="RESULT TODOFIX")
@pytest.mark.skipif(vendor_name == "hygon", reason="RuntimeError")
@pytest.mark.skipif(vendor_name == "mthreads", reason="Torch < 2.7")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="TypeError")
@pytest.mark.flash_attn_varlen_func
def test_perf_flash_attn_varlen_func():
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    bench = FlashAttnVarlenBenchmark(
        op_name="flash_attn_varlen_func",
        torch_op=flash_attn_varlen_func,
        dtypes=[torch.float16],
    )
    bench.set_gems(flag_gems.ops.flash_attn_varlen_func)
    bench.run()


class GetSchedulerMetadataBenchmark(GenericBenchmark):
    """
    benchmark for get_scheduler_metadata
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (8, 8, 1024, 16, 4, 128, 128),
            (32, 32, 512, 8, 8, 64, 64),
            (256, 256, 2048, 32, 32, 128, 128),
            (512, 512, 4096, 32, 8, 128, 128),
            (1024, 1024, 8192, 64, 16, 128, 128),
        ]

    def set_more_shapes(self):
        return None


@pytest.mark.get_scheduler_metadata
def test_perf_get_scheduler_metadata():
    try:
        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
        from vllm.vllm_flash_attn.flash_attn_interface import (
            get_scheduler_metadata as vllm_get_scheduler_metadata,
        )
    except ImportError:
        pytest.skip("vllm is not available, skipping performance test")

    def input_kwargs(shape, dtype, device):
        (
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            num_heads_q,
            num_heads_kv,
            headdim,
            headdim_v,
        ) = shape
        cache_seqlens = torch.randint(
            1, max_seqlen_k + 1, (batch_size,), dtype=torch.int32, device=device
        )

        yield (
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            num_heads_q,
            num_heads_kv,
            headdim,
            cache_seqlens,
            dtype,  # qkv_dtype
            headdim_v,  # headdim_v
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k_new
            None,  # cache_leftpad
            None,  # page_size
            0,  # max_seqlen_k_new
            False,  # causal
            (-1, -1),  # window_size
            False,  # has_softcap
            0,  # num_splits
            None,  # pack_gqa
            0,  # sm_margin
        )

    def flaggems_wrapper(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        cache_seqlens,
        qkv_dtype=torch.bfloat16,
        headdim_v=None,
        cu_seqlens_q=None,
        cu_seqlens_k_new=None,
        cache_leftpad=None,
        page_size=None,
        max_seqlen_k_new=0,
        causal=False,
        window_size=(-1, -1),
        has_softcap=False,
        num_splits=0,
        pack_gqa=None,
        sm_margin=0,
    ):
        return flag_gems.ops.get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads=num_heads_q,
            num_heads_k=num_heads_kv,
            headdim=headdim,
            headdim_v=headdim_v or headdim,
            qkv_dtype=qkv_dtype,
            seqused_k=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            cu_seqlens_k_new=cu_seqlens_k_new,
            seqused_q=None,
            leftpad_k=cache_leftpad,
            page_size=page_size,
            max_seqlen_k_new=max_seqlen_k_new,
            is_causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            has_softcap=has_softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )

    bench = GetSchedulerMetadataBenchmark(
        op_name="get_scheduler_metadata",
        input_fn=input_kwargs,
        torch_op=vllm_get_scheduler_metadata,
        dtypes=[
            torch.float16,
            torch.bfloat16,
        ],
    )
    bench.set_gems(flaggems_wrapper)
    bench.run()
