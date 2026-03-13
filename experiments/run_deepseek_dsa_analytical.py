#!/usr/bin/env python3
"""DeepSeek-V3 DSA Decode: Analytical per-layer timing breakdown.

Computes execution time for one decode step through 61 layers using
the correct DSA pipeline structure:

  Block A (Q compute)  ─┐
                        ├─ run in parallel ─→ Block C (Attention + Output)
  Block B (KV index)   ─┘

Block A: Q projection (compute the query for this decode token)
  - q_down_proj: hidden → q_lora_rank
  - q_up_proj:   q_lora_rank → H * (nope + rope)
  - RoPE on Q rope portion

Block B: KV indexing pipeline (find which KV entries to attend to)
  - Read indexer K cache from HBM (FP8, all seq_len tokens)
  - Indexer matmul: Q_compressed × indexer_K^T (score all tokens)
  - Top-k selection (pick 2048 tokens)
  - Fetch full MLA KV for selected + sliding window tokens (gather)

Block C: Attention compute (runs after both A and B complete)
  - kv_up_proj: decompress fetched KV latents → full K, V
  - Attention: Q × K^T → softmax → × V (on 2560 tokens only)
  - o_proj: attention output → hidden

Uses real H100 profiling data for GEMM times and IO bandwidth.
"""

import os
import pandas as pd
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILING_DIR = os.path.join(_ROOT, "data/profiling/compute/h100/deepseek_DeepSeek-V3")

# ── DeepSeek-V3 config ──────────────────────────────────────────────
NUM_LAYERS = 61
HIDDEN_SIZE = 7168
NUM_HEADS = 128
KV_LORA_RANK = 512
Q_LORA_RANK = 1536
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128

# Derived
Q_TOTAL_DIM = NUM_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)  # 128*192 = 24576
KV_UP_OUT_DIM = NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)       # 128*256 = 32768
O_PROJ_IN_DIM = NUM_HEADS * V_HEAD_DIM                             # 128*128 = 16384

# MLA KV cache: per token per layer
MLA_KV_BYTES_PER_TOKEN = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2  # 1152 bytes (FP16)

# DSA config
DSA_SELECTED_TOKENS = 2048
DSA_SLIDING_WINDOW = 512
DSA_ATTENDED = DSA_SELECTED_TOKENS + DSA_SLIDING_WINDOW  # 2560

# Indexer K: FP8, kv_lora_rank dims per token
INDEXER_K_BYTES_PER_TOKEN = KV_LORA_RANK * 1  # 512 bytes (FP8)

BYTES_PER_PARAM = 2  # FP16 weights

# ── H100 specs (from profiling) ────────────────────────────────────
# HBM: ~1350 GB/s at large sizes, but flat ~0.015 ms floor below 16 MB
# PCIe: ~51 GB/s
# FP16 TFLOPS: 989.5 (H100 SXM spec)
H100_FP16_TFLOPS = 989.5
H100_FP8_TFLOPS = 1979.0


def gemm_flops(m, k, n):
    return 2.0 * m * k * n


def gemm_weight_bytes(k, n):
    return k * n * BYTES_PER_PARAM


def load_profiling_data():
    attn = pd.read_csv(os.path.join(PROFILING_DIR, "attention.csv"))
    mlp = pd.read_csv(os.path.join(PROFILING_DIR, "mlp.csv"))
    io = pd.read_csv(os.path.join(PROFILING_DIR, "io.csv"))
    return attn, mlp, io


def get_profiled_time(df, num_tokens, col):
    """Get profiled median time for exact or nearest token count."""
    row = df[df["num_tokens"] == num_tokens]
    if len(row) > 0:
        return row.iloc[0][col]
    # Interpolate
    below = df[df["num_tokens"] <= num_tokens].sort_values("num_tokens")
    above = df[df["num_tokens"] >= num_tokens].sort_values("num_tokens")
    if len(below) == 0:
        return above.iloc[0][col]
    if len(above) == 0:
        return below.iloc[-1][col]
    lo = below.iloc[-1]
    hi = above.iloc[0]
    frac = (num_tokens - lo["num_tokens"]) / (hi["num_tokens"] - lo["num_tokens"])
    return lo[col] * (1 - frac) + hi[col] * frac


def hbm_read_time_ms(io_df, size_bytes):
    """Interpolate HBM read time from profiled data, respecting latency floor."""
    hbm = io_df[io_df["transfer_type"] == "hbm_read"].sort_values("size_bytes")
    size_mb = size_bytes / (1024 ** 2)

    # Exact or interpolate from profiled data
    below = hbm[hbm["size_bytes"] <= size_bytes]
    above = hbm[hbm["size_bytes"] >= size_bytes]

    if len(below) == 0:
        return above.iloc[0]["latency_ms"]
    if len(above) == 0:
        # Extrapolate using largest measured bandwidth
        bw = hbm.iloc[-1]["bandwidth_gb_per_s"]
        return (size_bytes / (bw * 1024**3)) * 1e3

    lo = below.iloc[-1]
    hi = above.iloc[0]

    if lo["size_bytes"] == hi["size_bytes"]:
        return lo["latency_ms"]

    frac = (size_bytes - lo["size_bytes"]) / (hi["size_bytes"] - lo["size_bytes"])
    return lo["latency_ms"] * (1 - frac) + hi["latency_ms"] * frac


def analyze_dsa_decode(seq_len, attn_df, mlp_df, io_df):
    """Analytical DSA decode timing for one layer at BS=1."""

    # ═══════════════════════════════════════════════════════════════
    # BLOCK A: Q Projection (1 decode token)
    # ═══════════════════════════════════════════════════════════════
    # These are weight-matrix multiplies at BS=1 → memory-bound (loading weights)
    q_down_ms = get_profiled_time(attn_df, 1, "time_stats.mla_q_down_proj.median")
    q_up_ms = get_profiled_time(attn_df, 1, "time_stats.mla_q_up_proj.median")
    rope_ms = get_profiled_time(attn_df, 1, "time_stats.mla_rope.median")

    q_down_weight_mb = gemm_weight_bytes(HIDDEN_SIZE, Q_LORA_RANK) / (1024**2)
    q_up_weight_mb = gemm_weight_bytes(Q_LORA_RANK, Q_TOTAL_DIM) / (1024**2)

    block_a_ms = q_down_ms + q_up_ms + rope_ms

    # ═══════════════════════════════════════════════════════════════
    # BLOCK B: KV Indexing Pipeline
    # ═══════════════════════════════════════════════════════════════

    # B1: Read indexer K cache from HBM (FP8, all tokens)
    indexer_k_bytes = INDEXER_K_BYTES_PER_TOKEN * seq_len
    indexer_k_mb = indexer_k_bytes / (1024**2)
    indexer_read_ms = hbm_read_time_ms(io_df, indexer_k_bytes)

    # B2: Indexer matmul (FP8): [1, 512] × [512, seq_len] → [1, seq_len]
    #     At BS=1 this is entirely memory-bound — time is dominated by
    #     reading the indexer K cache, which we already counted above.
    indexer_flops = gemm_flops(1, KV_LORA_RANK, seq_len)
    indexer_compute_ms = (indexer_flops / 1e9) / (H100_FP8_TFLOPS * 1024)  # negligible

    # B3: Top-k selection (2048 from seq_len)
    #     GPU parallel partial sort. Not profiled; estimate from GPU sort benchmarks.
    #     H100 can sort ~1B elements/s for top-k. At 128K: ~0.13 ms, at 1M: ~1 ms
    topk_ms = seq_len / 1e6 * 1.0  # rough: ~1 ms per 1M elements

    # B4: Fetch full MLA KV for attended tokens (scattered gather from KV cache)
    fetch_kv_bytes = MLA_KV_BYTES_PER_TOKEN * DSA_ATTENDED
    fetch_kv_mb = fetch_kv_bytes / (1024**2)
    # Scattered gather — use profiled HBM time (dominated by latency floor at 2.8 MB)
    fetch_kv_ms = hbm_read_time_ms(io_df, fetch_kv_bytes)

    block_b_ms = indexer_read_ms + indexer_compute_ms + topk_ms + fetch_kv_ms

    # ═══════════════════════════════════════════════════════════════
    # BLOCK C: Attention Compute (after max(A, B))
    # ═══════════════════════════════════════════════════════════════

    # C1: KV decompression (kv_up_proj on 2560 fetched tokens)
    #     [2560, 512] × [512, 32768] — compute-bound at this size
    kv_up_ms = get_profiled_time(attn_df, DSA_ATTENDED, "time_stats.mla_kv_up_proj.median")
    kv_up_flops = gemm_flops(DSA_ATTENDED, KV_LORA_RANK, KV_UP_OUT_DIM)

    # C2: Attention core (1 query vs 2560 KV entries, 128 heads)
    #     Memory-bound: reading decompressed KV from HBM
    #     Decompressed KV size: 2560 * 32768 * 2 = 160 MB
    decompressed_kv_bytes = DSA_ATTENDED * KV_UP_OUT_DIM * BYTES_PER_PARAM
    decompressed_kv_mb = decompressed_kv_bytes / (1024**2)
    attn_core_ms = hbm_read_time_ms(io_df, decompressed_kv_bytes)

    # C3: Output projection (on 1 decode token)
    #     [1, 16384] × [16384, 7168] — memory-bound at BS=1
    o_proj_ms = get_profiled_time(attn_df, 1, "time_stats.mla_o_proj.median")
    o_proj_weight_mb = gemm_weight_bytes(O_PROJ_IN_DIM, HIDDEN_SIZE) / (1024**2)

    # C4: Residual
    residual_ms = get_profiled_time(attn_df, 1, "time_stats.mla_block_residual.median")

    block_c_ms = kv_up_ms + attn_core_ms + o_proj_ms + residual_ms

    # ═══════════════════════════════════════════════════════════════
    # MoE BLOCK (sequential after attention)
    # ═══════════════════════════════════════════════════════════════
    moe_norm_ms = get_profiled_time(mlp_df, 1, "time_stats.moe_block_norm.median")
    moe_router_ms = (
        get_profiled_time(mlp_df, 1, "time_stats.moe_router_gate.median")
        + get_profiled_time(mlp_df, 1, "time_stats.moe_router_softmax.median")
        + get_profiled_time(mlp_df, 1, "time_stats.moe_router_topk.median")
    )
    moe_dispatch_ms = get_profiled_time(mlp_df, 1, "time_stats.moe_expert_dispatch.median")
    moe_gemm_ms = get_profiled_time(mlp_df, 1, "time_stats.moe_expert_gemm.median")
    moe_combine_ms = get_profiled_time(mlp_df, 1, "time_stats.moe_expert_combine.median")
    moe_shared_ms = get_profiled_time(mlp_df, 1, "time_stats.moe_shared_expert.median")
    moe_residual_ms = get_profiled_time(mlp_df, 1, "time_stats.moe_block_residual.median")

    moe_total_ms = (moe_norm_ms + moe_router_ms + moe_dispatch_ms +
                    moe_gemm_ms + moe_combine_ms + moe_shared_ms + moe_residual_ms)

    # ═══════════════════════════════════════════════════════════════
    # TOTAL per layer
    # ═══════════════════════════════════════════════════════════════
    parallel_ab_ms = max(block_a_ms, block_b_ms)
    attention_total_ms = parallel_ab_ms + block_c_ms
    layer_total_ms = attention_total_ms + moe_total_ms

    return {
        "seq_len": seq_len,
        # Block A
        "q_down_ms": q_down_ms,
        "q_up_ms": q_up_ms,
        "rope_ms": rope_ms,
        "block_a_ms": block_a_ms,
        "q_down_weight_mb": q_down_weight_mb,
        "q_up_weight_mb": q_up_weight_mb,
        # Block B
        "indexer_k_mb": indexer_k_mb,
        "indexer_read_ms": indexer_read_ms,
        "indexer_compute_ms": indexer_compute_ms,
        "topk_ms": topk_ms,
        "fetch_kv_mb": fetch_kv_mb,
        "fetch_kv_ms": fetch_kv_ms,
        "block_b_ms": block_b_ms,
        # Block C
        "kv_up_ms": kv_up_ms,
        "kv_up_gflops": kv_up_flops / 1e9,
        "decompressed_kv_mb": decompressed_kv_mb,
        "attn_core_ms": attn_core_ms,
        "o_proj_ms": o_proj_ms,
        "o_proj_weight_mb": o_proj_weight_mb,
        "residual_ms": residual_ms,
        "block_c_ms": block_c_ms,
        # MoE
        "moe_norm_ms": moe_norm_ms,
        "moe_router_ms": moe_router_ms,
        "moe_dispatch_ms": moe_dispatch_ms,
        "moe_gemm_ms": moe_gemm_ms,
        "moe_combine_ms": moe_combine_ms,
        "moe_shared_ms": moe_shared_ms,
        "moe_residual_ms": moe_residual_ms,
        "moe_total_ms": moe_total_ms,
        # Totals
        "parallel_ab_ms": parallel_ab_ms,
        "attention_total_ms": attention_total_ms,
        "layer_total_ms": layer_total_ms,
        "all_layers_ms": layer_total_ms * NUM_LAYERS,
    }


def print_breakdown(r):
    sl = r["seq_len"]
    sl_label = f"{sl // 1024}K" if sl < 1024 * 1024 else f"{sl // (1024 * 1024)}M"

    print(f"\n{'=' * 72}")
    print(f" DSA Decode — 1 Layer — BS=1 — seq_len={sl_label} ({sl:,} tokens)")
    print(f"{'=' * 72}")

    print(f"\n  BLOCK A: Q Projection (1 token, parallel with B)")
    print(f"  ┌─────────────────────────────────────────────────────────────")
    print(f"  │ q_down_proj  [1,7168]×[7168,1536]   weights={r['q_down_weight_mb']:.1f}MB   {r['q_down_ms']:.4f} ms")
    print(f"  │ q_up_proj    [1,1536]×[1536,24576]  weights={r['q_up_weight_mb']:.1f}MB  {r['q_up_ms']:.4f} ms")
    print(f"  │ RoPE         element-wise                           {r['rope_ms']:.4f} ms")
    print(f"  └─── Block A total: {r['block_a_ms']:.4f} ms")

    print(f"\n  BLOCK B: KV Indexing Pipeline (parallel with A)")
    print(f"  ┌─────────────────────────────────────────────────────────────")
    print(f"  │ Read indexer K   FP8, {sl_label} tokens   {r['indexer_k_mb']:.1f} MB   {r['indexer_read_ms']:.4f} ms")
    print(f"  │ Indexer matmul   [1,512]×[512,{sl}]  FP8       {r['indexer_compute_ms']:.6f} ms")
    print(f"  │ Top-k select     {sl} → 2048                     {r['topk_ms']:.4f} ms")
    print(f"  │ Fetch MLA KV     2560 tokens (gather)  {r['fetch_kv_mb']:.1f} MB   {r['fetch_kv_ms']:.4f} ms")
    print(f"  └─── Block B total: {r['block_b_ms']:.4f} ms")

    print(f"\n  ── max(A, B) = {r['parallel_ab_ms']:.4f} ms ──")

    print(f"\n  BLOCK C: Attention + Output (after A∥B)")
    print(f"  ┌─────────────────────────────────────────────────────────────")
    print(f"  │ kv_up_proj   [2560,512]×[512,32768]  {r['kv_up_gflops']:.1f} GFLOPs  {r['kv_up_ms']:.4f} ms")
    print(f"  │ Attn core    1 Q × 2560 KV  read {r['decompressed_kv_mb']:.0f}MB   {r['attn_core_ms']:.4f} ms")
    print(f"  │ o_proj       [1,16384]×[16384,7168]  weights={r['o_proj_weight_mb']:.0f}MB  {r['o_proj_ms']:.4f} ms")
    print(f"  │ Residual add                                        {r['residual_ms']:.4f} ms")
    print(f"  └─── Block C total: {r['block_c_ms']:.4f} ms")

    print(f"\n  MoE BLOCK (sequential after attention)")
    print(f"  ┌─────────────────────────────────────────────────────────────")
    print(f"  │ Norm          {r['moe_norm_ms']:.4f} ms")
    print(f"  │ Router        gate+softmax+topk                     {r['moe_router_ms']:.4f} ms")
    print(f"  │ Dispatch                                            {r['moe_dispatch_ms']:.4f} ms")
    print(f"  │ Expert GEMM   8 experts × 3 GEMMs (grouped)         {r['moe_gemm_ms']:.4f} ms")
    print(f"  │ Combine                                             {r['moe_combine_ms']:.4f} ms")
    print(f"  │ Shared expert 1 dense expert                        {r['moe_shared_ms']:.4f} ms")
    print(f"  │ Residual                                            {r['moe_residual_ms']:.4f} ms")
    print(f"  └─── MoE total: {r['moe_total_ms']:.4f} ms")

    print(f"\n  ─── LAYER TOTAL ─────────────────────────────────────────────")
    print(f"  │ Attention path: max(A,B) + C = {r['parallel_ab_ms']:.4f} + {r['block_c_ms']:.4f} = {r['attention_total_ms']:.4f} ms")
    print(f"  │ MoE block:                                          {r['moe_total_ms']:.4f} ms")
    print(f"  │ Layer total:                                        {r['layer_total_ms']:.4f} ms")
    print(f"  │")
    print(f"  │ × {NUM_LAYERS} layers = {r['all_layers_ms']:.2f} ms ({r['all_layers_ms']/1000:.3f} s)")
    print(f"  └────────────────────────────────────────────────────────────")

    # Where does the time go?
    attn_pct = r['attention_total_ms'] / r['layer_total_ms'] * 100
    moe_pct = r['moe_total_ms'] / r['layer_total_ms'] * 100
    print(f"\n  Time breakdown: Attention {attn_pct:.1f}% | MoE {moe_pct:.1f}%")


def main():
    attn_df, mlp_df, io_df = load_profiling_data()

    print("DeepSeek-V3 DSA Decode: Analytical Per-Layer Breakdown")
    print(f"Model: 61 layers, hidden=7168, 128 heads, MLA (kv_lora=512, q_lora=1536)")
    print(f"DSA: top-{DSA_SELECTED_TOKENS} selected + {DSA_SLIDING_WINDOW} sliding window = {DSA_ATTENDED} attended")
    print(f"Hardware: H100 (profiled), single GPU, BS=1")

    seq_lens = [4096, 32768, 128 * 1024, 512 * 1024, 1024 * 1024]
    results = []

    for sl in seq_lens:
        r = analyze_dsa_decode(sl, attn_df, mlp_df, io_df)
        results.append(r)
        print_breakdown(r)

    # Summary table
    print(f"\n\n{'=' * 90}")
    print(f" SUMMARY: DSA Decode Timing (BS=1, single H100, {NUM_LAYERS} layers)")
    print(f"{'=' * 90}")
    print(f"{'Seq Len':>10s}  {'Block A':>10s}  {'Block B':>10s}  {'max(A,B)':>10s}  "
          f"{'Block C':>10s}  {'MoE':>10s}  {'Layer':>10s}  {'61 Layers':>12s}")
    print(f"{' ':>10s}  {'Q proj':>10s}  {'KV index':>10s}  {'parallel':>10s}  "
          f"{'Attn+O':>10s}  {'experts':>10s}  {'total':>10s}  {'total':>12s}")
    print("-" * 90)
    for r in results:
        sl = r["seq_len"]
        sl_label = f"{sl // 1024}K" if sl < 1024 * 1024 else f"{sl // (1024 * 1024)}M"
        total_s = r["all_layers_ms"] / 1000
        print(f"{sl_label:>10s}  {r['block_a_ms']:>9.3f}ms  {r['block_b_ms']:>9.3f}ms  "
              f"{r['parallel_ab_ms']:>9.3f}ms  {r['block_c_ms']:>9.3f}ms  "
              f"{r['moe_total_ms']:>9.3f}ms  {r['layer_total_ms']:>9.3f}ms  "
              f"{total_s:>10.3f}s")
    print("-" * 90)


if __name__ == "__main__":
    main()
