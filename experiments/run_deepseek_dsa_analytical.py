#!/usr/bin/env python3
"""DeepSeek-V3 DSA decode per-layer timing — profiled compute + analytical IO.

Prints:
  1. Per-layer breakdown at each seq_len (4K, 32K, 128K, 512K, 1M) for
     both DSA modes (all-in-memory vs. offloading).
  2. Summary table comparing per-layer and 61-layer totals across modes.
  3. Max-batch-size comparison on an 80 GB H100 (weights ~60 GB).
  4. TPOT (time per output token) comparison between offload and HBM.
"""

import argparse
import json
import os

from experiments.dsa_layer_analyzer import (
    analyze_layer,
    compute_max_batch,
    run_sweep,
)
from experiments.dsa_timing_model import (
    DSA_ATTENDED,
    DSA_SELECTED_TOKENS,
    DSA_SLIDING_WINDOW,
    EP,
    H100_HBM_FLOOR_MS,
    H100_HBM_PEAK_GBS,
    H100_PCIE_FLOOR_MS,
    H100_PCIE_PEAK_GBS,
    HIDDEN_SIZE,
    KV_LORA_RANK,
    NUM_HEADS,
    NUM_LAYERS,
    NUM_ROUTED_EXPERTS,
    Q_LORA_RANK,
    load_profiles,
)

SEQ_LENS = [4096, 32768, 128 * 1024, 512 * 1024, 1024 * 1024]
MODES = ("hbm", "offload")


def sl_label(sl: int) -> str:
    return f"{sl // 1024}K" if sl < 1024 * 1024 else f"{sl // (1024 * 1024)}M"


def fmt_src(src: str) -> str:
    return {"profiled": "P", "analytical": "A",
            "kernel-floor": "K", "analytical-hbm": "A[H]",
            "analytical-offload": "A[P]"}.get(src, src[:3])


def print_breakdown(r: dict) -> None:
    sl = r["seq_len"]
    mode = r["mode"]
    src = r["source"]
    mode_label = "all-in-memory (HBM)" if mode == "hbm" else "offload (PCIe)"

    print(f"\n{'=' * 78}")
    print(f" DSA Decode — 1 Layer — BS=1 — seq_len={sl_label(sl)} ({sl:,} tokens)")
    print(f" Mode: {mode_label}")
    print(f"{'=' * 78}")

    print("\n  BLOCK A — Q projection  (parallel with B)")
    print("  ┌──────────────────────────────────────────────────────────────────")
    print(f"  │ pre-norm         [{fmt_src(src['pre_norm'])}]                       {r['pre_norm_ms']:.4f} ms")
    print(f"  │ q_down_proj      [{fmt_src(src['q_down'])}]  [1,7168]×[7168,1536]    {r['q_down_ms']:.4f} ms")
    print(f"  │ q_up_proj        [{fmt_src(src['q_up'])}]  [1,1536]×[1536,24576]   {r['q_up_ms']:.4f} ms")
    print(f"  │ RoPE             [{fmt_src(src['rope'])}]                       {r['rope_ms']:.4f} ms")
    print(f"  └── Block A total:                                {r['block_a_ms']:.4f} ms")

    print("\n  BLOCK B — KV indexing  (parallel with A)")
    print("  ┌──────────────────────────────────────────────────────────────────")
    print(f"  │ indexer K read   [{fmt_src(src['indexer_read'])}]  {r['indexer_k_mb']:>10.2f} MB          {r['indexer_read_ms']:.4f} ms")
    print(f"  │ indexer matmul   [{fmt_src(src['indexer_compute'])}]  FP8 [1,512]×[512,{sl}]  {r['indexer_compute_ms']:.4f} ms")
    print(f"  │ top-k select     [{fmt_src(src['topk'])}]  {sl} → 2048             {r['topk_ms']:.4f} ms")
    print(f"  │ fetch MLA KV     [{fmt_src(src['fetch_kv'])}]  {r['fetch_kv_mb']:>10.2f} MB gather    {r['fetch_kv_ms']:.4f} ms")
    print(f"  └── Block B total:                                {r['block_b_ms']:.4f} ms")

    print(f"\n  ── max(A, B) = {r['parallel_ab_ms']:.4f} ms ──")

    print("\n  BLOCK C — Attention + output")
    print("  ┌──────────────────────────────────────────────────────────────────")
    print(f"  │ kv_up_proj       [{fmt_src(src['kv_up'])}]  [2560,512]×[512,32768]   {r['kv_up_ms']:.4f} ms")
    print(f"  │ attn core        [{fmt_src(src['attn_core'])}]  1 Q × 2560 KV × 128H  {r['attn_core_ms']:.4f} ms")
    print(f"  │                         decompressed KV = {r['decompressed_kv_mb']:.1f} MB")
    print(f"  │ o_proj           [{fmt_src(src['o_proj'])}]  [1,16384]×[16384,7168]  {r['o_proj_ms']:.4f} ms")
    print(f"  │ residual add     [{fmt_src(src['residual'])}]                       {r['residual_ms']:.4f} ms")
    print(f"  └── Block C total:                                {r['block_c_ms']:.4f} ms")

    print("\n  MoE BLOCK  (EP=8, 32 experts resident per GPU)")
    print("  ┌──────────────────────────────────────────────────────────────────")
    print(f"  │ norm             [{fmt_src(src['moe_norm'])}]                       {r['moe_norm_ms']:.4f} ms")
    print(f"  │ router gate      [{fmt_src(src['router_gate'])}]  [1,7168]×[7168,256]     {r['router_gate_ms']:.4f} ms")
    print(f"  │ router softmax   [{fmt_src(src['router_softmax'])}]                       {r['router_softmax_ms']:.4f} ms")
    print(f"  │ router top-k     [{fmt_src(src['router_topk'])}]                       {r['router_topk_ms']:.4f} ms")
    print(f"  │ EP dispatch      [{fmt_src(src['ep_dispatch'])}]  8 NVLink messages      {r['ep_dispatch_ms']:.4f} ms")
    print(f"  │ expert GEMM      [{fmt_src(src['expert_gemm'])}]  1 expert × 84 MB / HBM {r['expert_gemm_ms']:.4f} ms")
    print(f"  │ EP combine       [{fmt_src(src['ep_combine'])}]  8 NVLink messages      {r['ep_combine_ms']:.4f} ms")
    print(f"  │ shared expert    [{fmt_src(src['shared_expert'])}]  84 MB / HBM            {r['shared_expert_ms']:.4f} ms")
    print(f"  │ residual         [{fmt_src(src['moe_residual'])}]                       {r['moe_residual_ms']:.4f} ms")
    print(f"  └── MoE total:                                    {r['moe_total_ms']:.4f} ms")

    print("\n  ── LAYER TOTAL ──────────────────────────────────────────────────")
    print(f"  │ Attention path:  max(A,B) + C = {r['parallel_ab_ms']:.4f} + {r['block_c_ms']:.4f} = {r['attention_total_ms']:.4f} ms")
    print(f"  │ MoE:                                            {r['moe_total_ms']:.4f} ms")
    print(f"  │ Layer total:                                    {r['layer_total_ms']:.4f} ms")
    print(f"  │ × {NUM_LAYERS} layers:                                   {r['all_layers_ms']:.2f} ms"
          f"  ({r['all_layers_ms']/1000:.3f} s)")
    print("  └────────────────────────────────────────────────────────────────")


def print_mode_comparison(results: dict) -> None:
    print(f"\n\n{'=' * 96}")
    print(" SUMMARY — DSA decode per-layer timing, BS=1, H100 SXM")
    print(" Source legend:  profiled compute CSV (num_tokens lookup)  +  analytical IO")
    print(f"{'=' * 96}")
    print(f"{'seq_len':>8}  {'mode':>8}  "
          f"{'A(ms)':>8}  {'B(ms)':>8}  {'max(A,B)':>9}  "
          f"{'C(ms)':>8}  {'MoE(ms)':>9}  {'layer(ms)':>11}  {'61L(ms)':>10}  {'TPOT(ms)':>10}")
    print("-" * 96)
    for sl in SEQ_LENS:
        for mode in MODES:
            r = results[(mode, sl)]
            print(f"{sl_label(sl):>8}  {mode:>8}  "
                  f"{r['block_a_ms']:>8.4f}  {r['block_b_ms']:>8.4f}  {r['parallel_ab_ms']:>9.4f}  "
                  f"{r['block_c_ms']:>8.4f}  {r['moe_total_ms']:>9.4f}  {r['layer_total_ms']:>11.4f}  "
                  f"{r['all_layers_ms']:>10.2f}  {r['all_layers_ms']:>10.2f}")
    print("-" * 96)


def print_tpot_comparison(results: dict) -> None:
    print(f"\n\n{'=' * 78}")
    print(" TPOT COMPARISON  (time per output token = 61 layers × layer_time)")
    print(f"{'=' * 78}")
    print(f"{'seq_len':>10}  {'TPOT hbm (ms)':>16}  {'TPOT offload (ms)':>20}  "
          f"{'Δ (ms)':>10}  {'slowdown':>10}")
    print("-" * 78)
    for sl in SEQ_LENS:
        hbm = results[("hbm", sl)]["all_layers_ms"]
        off = results[("offload", sl)]["all_layers_ms"]
        delta = off - hbm
        slow = off / hbm
        print(f"{sl_label(sl):>10}  {hbm:>16.2f}  {off:>20.2f}  "
              f"{delta:>10.2f}  {slow:>9.2f}x")
    print("-" * 78)


def print_max_batch(results: dict) -> None:
    print(f"\n\n{'=' * 82}")
    print(" MAX CONCURRENT SEQUENCES — H100 80 GB, weights ≈ 60 GB, KV budget = 20 GB")
    print(f"{'=' * 82}")
    print(f"{'seq_len':>10}  {'mode':>10}  {'KV / seq':>12}  {'indexer / seq':>14}  "
          f"{'HBM / seq':>12}  {'max seqs':>10}")
    print("-" * 82)
    for sl in SEQ_LENS:
        for mode in MODES:
            b = compute_max_batch(mode, sl)
            max_s = "HBM-unbounded" if b["max_concurrent_seqs"] < 0 else f"{b['max_concurrent_seqs']}"
            print(f"{sl_label(sl):>10}  {mode:>10}  "
                  f"{b['per_seq_kv_mb']:>10.1f} MB  "
                  f"{b['per_seq_indexer_mb']:>12.1f} MB  "
                  f"{b['per_seq_hbm_mb']:>10.1f} MB  "
                  f"{max_s:>10}")
    print("-" * 82)
    print("  * offload mode keeps KV on CPU, so HBM capacity is not the bottleneck —")
    print("    concurrency is instead bounded by PCIe bandwidth per layer (see TPOT table).")


def print_header():
    t = load_profiles()
    print("DeepSeek-V3 DSA Decode — Per-Layer Timing (profiled compute + analytical IO)")
    print(f"Model: {NUM_LAYERS} layers | hidden={HIDDEN_SIZE} | heads={NUM_HEADS} | "
          f"MLA (kv_lora={KV_LORA_RANK}, q_lora={Q_LORA_RANK})")
    print(f"DSA:   top-{DSA_SELECTED_TOKENS} + sliding-{DSA_SLIDING_WINDOW} = "
          f"{DSA_ATTENDED} attended tokens")
    print(f"MoE:   {NUM_ROUTED_EXPERTS} routed experts, EP={EP} ({NUM_ROUTED_EXPERTS//EP}/GPU)")
    print(f"IO:    HBM peak {H100_HBM_PEAK_GBS:.0f} GB/s (floor {H100_HBM_FLOOR_MS:.3f} ms) | "
          f"PCIe Gen4 x16 {H100_PCIE_PEAK_GBS:.1f} GB/s (floor {H100_PCIE_FLOOR_MS:.3f} ms) | "
          f"NVLink 5 μs/msg")
    print(f"Profile: {len(t.attn)} attn rows (nt max={t.attn_max}), "
          f"{len(t.mlp)} mlp rows (nt max={t.mlp_max})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true",
                    help="Skip per-layer breakdowns, print only summary tables.")
    ap.add_argument("--json-out", type=str, default=None,
                    help="Dump raw results as JSON for downstream plotting.")
    args = ap.parse_args()

    print_header()
    results = run_sweep(SEQ_LENS, MODES)

    if not args.summary_only:
        for sl in SEQ_LENS:
            for mode in MODES:
                print_breakdown(results[(mode, sl)])

    print_mode_comparison(results)
    print_tpot_comparison(results)
    print_max_batch(results)

    if args.json_out:
        # Strip non-JSON-serializable bits
        serializable = {}
        for (mode, sl), r in results.items():
            serializable[f"{mode}_{sl}"] = {k: v for k, v in r.items()
                                             if k != "source"}
            serializable[f"{mode}_{sl}"]["source"] = r["source"]
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n[wrote {args.json_out}]")


if __name__ == "__main__":
    main()
