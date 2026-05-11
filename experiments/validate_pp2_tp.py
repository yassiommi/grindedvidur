#!/usr/bin/env python3
"""Validate DSA vs IndexCache at PP=4 TP={2,4} EP=8 with analytical compute.

EP=8 is the realistic shard (1 active expert per rank per token at BS=1
when 8 active out of 256 experts and 8 EP groups). EP=16 over-shards
(0.5 active per rank, half idle). With EP=8 + FP16, per-rank expert
weights are 21 GB/layer × layers_per_stage / 8 = 2.625 GB × layers.
At PP=2 (31 layers/stage) that's 81 GB — exceeds H100 80 GB. So we
bump to PP=4 (16 layers/stage, 42 GB expert weights per rank), which
fits comfortably with KV + IDX + non-expert weights.

Steps (everything hand-derived alongside the model):
  1. Per-rank memory at PP=4 TP=2 EP=8 and PP=4 TP=4 EP=8
  2. Per-piece per-layer cost hand-derived from HBM/PCIe bandwidth
  3. TP-scaled per-layer compute at TP=2 and TP=4
  4. Per-stage PP pipeline derivation (stage 0 producer/consumer; stages
     1..PP-1 with parallel IO buses)
  5. DSA vs IndexCache step time across (BS, sl, mode) corners
  6. Identity checks (model vs hand) at every step

Run:
    python -m experiments.validate_pp2_tp
"""

from __future__ import annotations

import math

from experiments.analytical_layer import (
    DSA_ATTENDED, DSA_SLIDING, DSA_TOPK,
    EXPERT_INTERMEDIATE, EXPERT_W_PER_LAYER_B,
    HBM_BW_GBPS, HBM_FLOOR_MS, HIDDEN, INDEXER_K_BYTES_PER_TOKEN,
    KV_LORA_RANK, MLA_KV_BYTES_PER_TOKEN,
    NON_EXPERT_W_PER_LAYER_B, NUM_HEADS, NUM_LAYERS, NUM_ROUTED_EXPERTS,
    PCIE_BW_GBPS, Q_LORA_RANK, QK_NOPE, QK_ROPE, V_HEAD,
    analytical_layer_F, analytical_layer_S,
    hbm_ms, io_ms, pcie_ms, per_rank_memory_bytes,
)

GB = 1024 ** 3
MB = 1024 ** 2


def hr(title: str):
    print()
    print("=" * 92)
    print(f" {title}")
    print("=" * 92)


def chk(name: str, got: float, want: float, tol_rel: float = 1e-9,
        tol_abs: float = 1e-9) -> bool:
    diff = abs(got - want)
    ok = diff <= max(tol_abs, tol_rel * max(abs(got), abs(want), 1.0))
    flag = "[OK ]" if ok else "[FAIL]"
    print(f"  {flag}  {name:<58}  model={got:>14.6f}  hand={want:>14.6f}")
    return ok


# ─────────────────────────────────────────────────────────────────────
# Schedule: producer/consumer within a stage, parallel buses across PP
# ─────────────────────────────────────────────────────────────────────
def pipe_within_stage(layer_costs: list, tp: int) -> float:
    """Producer/consumer schedule for one PP stage's layers.

    layer_costs: list of AnalyticalLayerCost (this stage's layers).
    tp: tensor parallelism degree (scales compute side, leaves IO).
    Returns the wall time when the stage's last compute ends.
    """
    if not layer_costs:
        return 0.0
    cum_io = 0.0
    end_comp = 0.0
    for c in layer_costs:
        cum_io += c.total_io_ms()
        comp = c.total_compute_ms(tp=tp)
        cmp_start = max(end_comp, cum_io)
        end_comp = cmp_start + comp
    return end_comp


def schedule_pp(layer_costs: list, pp: int, tp: int, n_layers: int = NUM_LAYERS) -> dict:
    """PP-aware schedule.

    Stage 0: producer/consumer through its layers (own IO bus).
    Stages 1..PP-1: independent IO buses prefetch their IO from t=0;
        compute starts at max(prev_end, stage_io_total).
    """
    layers_per_stage = math.ceil(n_layers / pp)
    stages = [layer_costs[i*layers_per_stage:(i+1)*layers_per_stage]
              for i in range(pp)]
    stages = [s for s in stages if s]
    assert sum(len(s) for s in stages) == len(layer_costs)

    # Stage 0
    end_comp = pipe_within_stage(stages[0], tp)
    cold = stages[0][0].total_io_ms() if stages[0] else 0.0
    stage_times = [{"layers": len(stages[0]),
                    "io_total": sum(c.total_io_ms() for c in stages[0]),
                    "compute_total": sum(c.total_compute_ms(tp=tp) for c in stages[0]),
                    "end": end_comp}]

    for k in range(1, len(stages)):
        stage = stages[k]
        stage_io = sum(c.total_io_ms() for c in stage)
        stage_cmp = sum(c.total_compute_ms(tp=tp) for c in stage)
        cmp_start = max(end_comp, stage_io)
        end_comp = cmp_start + stage_cmp
        stage_times.append({"layers": len(stage), "io_total": stage_io,
                            "compute_total": stage_cmp, "end": end_comp})

    total_io   = sum(c.total_io_ms() for c in layer_costs)
    total_comp = sum(c.total_compute_ms(tp=tp) for c in layer_costs)
    return {
        "total_ms": end_comp,
        "io_total_ms": total_io,
        "compute_total_ms": total_comp,
        "cold_ms": cold,
        "stages": stage_times,
    }


def build_pattern(n_layers: int, f_period: int,
                  seq_len: int, bs: int, mode: str, ep: int,
                  fp8: bool = False) -> list:
    """F:S:S:...:S layer cost list with the given f_period."""
    costs = []
    for i in range(n_layers):
        if i % f_period == 0:
            costs.append(analytical_layer_F(seq_len, bs, mode, ep, fp8=fp8))
        else:
            costs.append(analytical_layer_S(seq_len, bs, mode, ep, fp8=fp8))
    return costs


# ─────────────────────────────────────────────────────────────────────
def main():
    print("DSA vs IndexCache at PP=4, TP={2,4}, EP=8  (analytical, BS=1 sl=200K offload)")
    print("All numbers hand-derived from HBM/PCIe peak BW and FLOPS peak.\n")

    # ===== STEP 1: per-rank memory =====
    hr("STEP 1 — Per-rank memory (PP=4 TP=2/4 EP=8, BS=1 sl=200K)")
    print(f"  Constants:")
    print(f"    non-expert weights / layer  = {NON_EXPERT_W_PER_LAYER_B / MB:.2f} MB (FP16)")
    print(f"    expert weights / layer      = {EXPERT_W_PER_LAYER_B / GB:.2f} GB (FP16, 256 × 84 MB)")
    print(f"    total non-expert (61 lyr)   = {NUM_LAYERS * NON_EXPERT_W_PER_LAYER_B / GB:.2f} GB")
    print(f"    total expert     (61 lyr)   = {NUM_LAYERS * EXPERT_W_PER_LAYER_B / GB:.2f} GB")
    print(f"    total weights               = {NUM_LAYERS * (NON_EXPERT_W_PER_LAYER_B + EXPERT_W_PER_LAYER_B) / GB:.2f} GB")
    print()
    for pp, tp, ep, label in [(4, 2, 8, "PP=4 TP=2 EP=8"),
                              (4, 4, 8, "PP=4 TP=4 EP=8")]:
        m = per_rank_memory_bytes(pp, tp, ep, 200*1024, 1)
        layers = math.ceil(NUM_LAYERS / pp)
        kv_h  = layers * 1 * 200 * 1024 * MLA_KV_BYTES_PER_TOKEN
        idx_h = layers * 1 * 200 * 1024 * INDEXER_K_BYTES_PER_TOKEN
        nW_h  = layers * NON_EXPERT_W_PER_LAYER_B // tp
        eW_h  = layers * EXPERT_W_PER_LAYER_B // ep
        total_h = kv_h + idx_h + nW_h + eW_h
        print(f"  {label}  →  layers_per_stage = {layers}")
        chk(f"  {label} KV bytes",          m["kv"],           kv_h)
        chk(f"  {label} IDX bytes",         m["idx"],          idx_h)
        chk(f"  {label} non-expert W bytes", m["non_expert_w"], nW_h)
        chk(f"  {label} expert W bytes",    m["expert_w"],     eW_h)
        chk(f"  {label} total bytes",       m["total"],        total_h)
        print(f"    breakdown: KV {kv_h/GB:.2f} GB  IDX {idx_h/GB:.2f} GB  "
              f"nonW {nW_h/GB:.2f} GB  expW {eW_h/GB:.2f} GB  total {total_h/GB:.2f} GB")
        print(f"    fits 80 GB w/ 8 GB headroom? {'YES' if total_h/GB + 8 <= 80 else 'NO'}")
        print()

    # ===== STEP 2: per-piece layer cost hand-derivation =====
    hr("STEP 2 — Per-piece F-layer cost (BS=1 sl=200K offload, TP=1 baseline)")
    F = analytical_layer_F(200 * 1024, 1, "offload", ep=8)
    S = analytical_layer_S(200 * 1024, 1, "offload", ep=8)
    print(f"  F-layer pieces (from model):")
    print(f"    idx_io   = {F.idx_io_ms:>8.5f} ms   (100 MB / 51.5 GB/s PCIe)")
    print(f"    kv_io    = {F.kv_io_ms:>8.5f} ms   (2.81 MB / 51.5 GB/s PCIe)")
    print(f"    block_a  = {F.block_a_ms:>8.5f} ms")
    print(f"    block_c  = {F.block_c_ms:>8.5f} ms")
    print(f"    idx_comp = {F.idx_comp_ms:>8.5f} ms")
    print(f"    moe shardable = {F.moe_shardable_ms:>8.5f} ms")
    print(f"    moe expert_gemm = {F.moe_expert_gemm_ms:>8.5f} ms (EP=8, 32 local exp avg 1.0 unique/tok)")
    print(f"    moe ep_comms = {F.moe_ep_comms_ms:>8.5f} ms")
    print(f"    TOTAL T_compute (TP=1) = {F.total_compute_ms(tp=1):>8.5f} ms")
    print()
    print(f"  S-layer (same minus idx_io and idx_comp):")
    print(f"    idx_io   = {S.idx_io_ms:>8.5f} ms (zero)")
    print(f"    idx_comp = {S.idx_comp_ms:>8.5f} ms (zero)")
    print(f"    TOTAL T_compute (TP=1) = {S.total_compute_ms(tp=1):>8.5f} ms")

    # Hand-derive idx_io and kv_io
    idx_io_hand = 1 * 200*1024 * INDEXER_K_BYTES_PER_TOKEN / (PCIE_BW_GBPS * GB) * 1000
    kv_io_hand  = 1 * DSA_ATTENDED * MLA_KV_BYTES_PER_TOKEN / (PCIE_BW_GBPS * GB) * 1000
    print(f"\n  Hand:")
    print(f"    idx_io_hand = 1 × 200K × 512 B / (51.5 GB/s) × 1000 = {idx_io_hand:.5f} ms")
    print(f"    kv_io_hand  = 1 × 2560 × 1152 B / (51.5 GB/s) × 1000 = {kv_io_hand:.5f} ms")
    chk("F.idx_io_ms matches hand", F.idx_io_ms, idx_io_hand, tol_rel=1e-9)
    chk("F.kv_io_ms  matches hand", F.kv_io_ms,  kv_io_hand,  tol_rel=1e-9)

    # ===== STEP 3: TP-scaled per-layer compute =====
    hr("STEP 3 — TP scaling of per-layer compute")
    for tp in (1, 2, 4):
        tF = F.total_compute_ms(tp=tp)
        tS = S.total_compute_ms(tp=tp)
        print(f"  TP={tp}:  T_F_compute = {tF:.4f} ms   T_S_compute = {tS:.4f} ms")

    # Hand-derive TP=2 F compute:
    # T_compute = max(0, block_a/tp - io) + (block_c + idx_comp + moe_shardable)/tp
    #             + moe_expert_gemm + moe_ep_comms
    # With block_a/2 = 0.049, io_total = 1.95, → block_a fully hidden → 0
    # (block_c + idx_comp + moe_shardable) / 2 = (0.43 + 0.01 + 0.113)/2 ≈ 0.275
    # + expert_gemm + ep_comms (unchanged)
    block_a_tp2 = F.block_a_ms / 2
    io_F = F.total_io_ms()
    block_a_exp = max(0, block_a_tp2 - io_F)
    rest = (F.block_c_ms + F.idx_comp_ms + F.moe_shardable_ms) / 2
    hand_tF_tp2 = block_a_exp + rest + F.moe_expert_gemm_ms + F.moe_ep_comms_ms
    print(f"\n  Hand TP=2 F:  block_a_exposed=max(0, {block_a_tp2:.4f} - {io_F:.4f})={block_a_exp:.4f}")
    print(f"               + (block_c + idx_comp + moe_shardable)/2 = {rest:.4f}")
    print(f"               + expert_gemm + ep_comms = "
          f"{F.moe_expert_gemm_ms + F.moe_ep_comms_ms:.4f}")
    print(f"               = {hand_tF_tp2:.4f} ms")
    chk("T_F_compute TP=2 vs hand", F.total_compute_ms(tp=2), hand_tF_tp2, tol_rel=1e-12)

    # ===== STEP 4: PP=4 step time at TP=2 / TP=4 =====
    hr("STEP 4 — PP=4 step time for DSA and IndexCache F:S:S:S")

    sl, bs, mode = 200 * 1024, 1, "offload"
    print(f"  Workload: sl={sl//1024}K, BS={bs}, mode={mode}")
    print(f"  Pattern DSA: 61 F layers. IndexCache: F:S:S:S → 16 F, 45 S\n")

    for tp in (2, 4):
        print(f"  ── TP={tp} ──")
        dsa_costs = build_pattern(NUM_LAYERS, 1, sl, bs, mode, ep=8)
        ic_costs  = build_pattern(NUM_LAYERS, 4, sl, bs, mode, ep=8)

        dsa = schedule_pp(dsa_costs, pp=4, tp=tp)
        ic  = schedule_pp(ic_costs,  pp=4, tp=tp)

        for stage_idx, st in enumerate(dsa["stages"]):
            print(f"    DSA stage {stage_idx}: {st['layers']:>2} layers  "
                  f"io_total={st['io_total']:>7.3f}  "
                  f"compute_total={st['compute_total']:>7.3f}  "
                  f"end={st['end']:>7.3f}")
        print(f"    DSA total = {dsa['total_ms']:.4f} ms  "
              f"(io_total {dsa['io_total_ms']:.3f}, compute_total {dsa['compute_total_ms']:.3f})")
        print()
        for stage_idx, st in enumerate(ic["stages"]):
            print(f"    IC  stage {stage_idx}: {st['layers']:>2} layers  "
                  f"io_total={st['io_total']:>7.3f}  "
                  f"compute_total={st['compute_total']:>7.3f}  "
                  f"end={st['end']:>7.3f}")
        print(f"    IC  total = {ic['total_ms']:.4f} ms")
        print(f"    Speedup IC/DSA = {dsa['total_ms']/ic['total_ms']:.3f}x")
        print()

    # ===== STEP 5: Verify producer/consumer at stage 0 by hand =====
    hr("STEP 5 — Hand-verify stage 0 producer/consumer (DSA, TP=2)")
    layers = math.ceil(NUM_LAYERS / 4)  # 16 (PP=4)
    print(f"  Stage 0 has {layers} F-layers.")
    print(f"  Each layer:")
    print(f"    T_io      = {F.total_io_ms():.4f} ms")
    print(f"    T_compute = {F.total_compute_ms(tp=2):.4f} ms (TP=2)")
    print(f"\n  Walk:")
    cum_io = 0.0
    end_comp = 0.0
    for i in range(min(5, layers)):
        cum_io += F.total_io_ms()
        cmp_start = max(end_comp, cum_io)
        end_comp = cmp_start + F.total_compute_ms(tp=2)
        print(f"    layer {i}: cum_io={cum_io:>7.4f}, cmp_start={cmp_start:>7.4f}, "
              f"end_comp={end_comp:>7.4f}")
    print(f"    ... walk through all {layers} layers ...")
    # Run the full walk by hand
    cum_io = 0.0
    end_comp = 0.0
    for _ in range(layers):
        cum_io += F.total_io_ms()
        end_comp = max(end_comp, cum_io) + F.total_compute_ms(tp=2)
    print(f"    layer {layers-1} end_comp = {end_comp:.4f} ms")
    # Compare to model
    dsa_costs = build_pattern(NUM_LAYERS, 1, sl, bs, mode, ep=8)
    dsa_sched = schedule_pp(dsa_costs, pp=4, tp=2)
    chk("Stage 0 end_compute (DSA TP=2) vs hand",
        dsa_sched["stages"][0]["end"], end_comp, tol_rel=1e-9)

    # ===== STEP 6: Fine-grained sweep across corners =====
    hr("STEP 6 — DSA vs IndexCache sweep across feasible corners (PP=4 TP=2/4 EP=8)")

    scenarios = [
        # (label, sl, bs)
        ("4K   / BS=1", 4 * 1024, 1),
        ("32K  / BS=1", 32 * 1024, 1),
        ("128K / BS=1", 128 * 1024, 1),
        ("200K / BS=1", 200 * 1024, 1),
        ("512K / BS=1", 512 * 1024, 1),
        ("1M   / BS=1", 1024 * 1024, 1),
        ("2M   / BS=1", 2 * 1024 * 1024, 1),
        ("4M   / BS=1", 4 * 1024 * 1024, 1),
        ("4K   / BS=8", 4 * 1024, 8),
        ("32K  / BS=8", 32 * 1024, 8),
        ("128K / BS=8", 128 * 1024, 8),
        ("200K / BS=8", 200 * 1024, 8),
        ("512K / BS=8", 512 * 1024, 8),
        ("4K   / BS=32", 4 * 1024, 32),
        ("32K  / BS=32", 32 * 1024, 32),
        ("128K / BS=32", 128 * 1024, 32),
        ("200K / BS=32", 200 * 1024, 32),
        ("512K / BS=32", 512 * 1024, 32),
    ]

    for tp in (2, 4):
        ep = 8
        print(f"\n  ── PP=4 TP={tp} EP={ep}  (offload mode) ──")
        print(f"  TPOT = step time = time per output token, per user.")
        print(f"  Throughput = BS × 1000 / step_ms  (aggregate tokens/sec across all BS users on this cluster).\n")
        print(f"  {'scenario':>14}  {'mem GB':>7} {'fit?':>5}  "
              f"{'DSA step':>9} {'IC step':>9} {'speedup':>8}  "
              f"{'DSA TPOT':>9} {'IC TPOT':>9}  "
              f"{'DSA tok/s':>10} {'IC tok/s':>10}  "
              f"{'DSA stage0':>11} {'IC stage0':>10}")
        print("  " + "-" * 137)
        for label, sl_i, bs_i in scenarios:
            m = per_rank_memory_bytes(4, tp, ep, sl_i, bs_i)
            fits = m["total"] / GB + 8 <= 80
            dsa_costs = build_pattern(NUM_LAYERS, 1, sl_i, bs_i, "offload", ep=ep)
            ic_costs  = build_pattern(NUM_LAYERS, 4, sl_i, bs_i, "offload", ep=ep)
            dsa = schedule_pp(dsa_costs, pp=4, tp=tp)
            ic  = schedule_pp(ic_costs,  pp=4, tp=tp)
            dsa_tput = bs_i * 1000.0 / dsa["total_ms"]
            ic_tput  = bs_i * 1000.0 / ic["total_ms"]
            print(f"  {label:>14}  {m['total']/GB:>6.2f}   {'YES' if fits else 'NO':>4}   "
                  f"{dsa['total_ms']:>8.3f}  {ic['total_ms']:>8.3f}  "
                  f"{dsa['total_ms']/ic['total_ms']:>7.2f}x  "
                  f"{dsa['total_ms']:>8.3f}  {ic['total_ms']:>8.3f}  "
                  f"{dsa_tput:>9.1f}  {ic_tput:>9.1f}  "
                  f"{dsa['stages'][0]['end']:>10.3f}  "
                  f"{ic['stages'][0]['end']:>10.3f}")

    # HBM mode — full scenario list (same memory feasibility as offload).
    hr("STEP 7 — Same scenarios but HBM mode (compute-bound, IO is cheap)")
    for tp in (2, 4):
        ep = 8
        print(f"\n  ── PP=4 TP={tp} EP={ep}  (HBM mode) ──")
        print(f"  {'scenario':>14}  {'mem GB':>7} {'fit?':>5}  "
              f"{'DSA step':>9} {'IC step':>9} {'speedup':>8}  "
              f"{'DSA TPOT':>9} {'IC TPOT':>9}  "
              f"{'DSA tok/s':>10} {'IC tok/s':>10}")
        print("  " + "-" * 110)
        for label, sl_i, bs_i in scenarios:
            m = per_rank_memory_bytes(4, tp, ep, sl_i, bs_i)
            fits = m["total"] / GB + 8 <= 80
            dsa_costs = build_pattern(NUM_LAYERS, 1, sl_i, bs_i, "hbm", ep=ep)
            ic_costs  = build_pattern(NUM_LAYERS, 4, sl_i, bs_i, "hbm", ep=ep)
            dsa = schedule_pp(dsa_costs, pp=4, tp=tp)
            ic  = schedule_pp(ic_costs,  pp=4, tp=tp)
            dsa_tput = bs_i * 1000.0 / dsa["total_ms"]
            ic_tput  = bs_i * 1000.0 / ic["total_ms"]
            print(f"  {label:>14}  {m['total']/GB:>6.2f}   {'YES' if fits else 'NO':>4}   "
                  f"{dsa['total_ms']:>8.3f}  {ic['total_ms']:>8.3f}  "
                  f"{dsa['total_ms']/ic['total_ms']:>7.2f}x  "
                  f"{dsa['total_ms']:>8.3f}  {ic['total_ms']:>8.3f}  "
                  f"{dsa_tput:>9.1f}  {ic_tput:>9.1f}")

    # ===== STEP 8: FP8 sweep (matches DeepSeek-V3.2-Exp production) =====
    hr("STEP 8 — FP8 weights + FP8 KV cache (PP=4 EP=8, both modes)")
    print("  Weights and KV cache at 1 byte/elem. Indexer K is FP8 in both")
    print("  paths (unchanged). Halves per-rank memory and per-layer compute.")
    print()

    # Hand-verify a key memory number at PP=4 TP=4 EP=8 FP8 BS=1 sl=200K.
    m_hand_fp8 = {
        "kv":  16 * 1 * 200*1024 * (KV_LORA_RANK + QK_ROPE) * 1,  # FP8 KV
        "idx": 16 * 1 * 200*1024 * INDEXER_K_BYTES_PER_TOKEN,
        "nW":  16 * (NON_EXPERT_W_PER_LAYER_B // 2) // 4,
        "eW":  16 * (EXPERT_W_PER_LAYER_B // 2) // 8,
    }
    m_hand_fp8["total"] = sum(m_hand_fp8.values())
    m_model_fp8 = per_rank_memory_bytes(4, 4, 8, 200*1024, 1, fp8=True)
    print("  Hand-verify FP8 memory at PP=4 TP=4 EP=8 BS=1 sl=200K:")
    chk("  FP8 KV bytes",          m_model_fp8["kv"],           m_hand_fp8["kv"])
    chk("  FP8 IDX bytes",         m_model_fp8["idx"],          m_hand_fp8["idx"])
    chk("  FP8 non-expert W bytes", m_model_fp8["non_expert_w"], m_hand_fp8["nW"])
    chk("  FP8 expert W bytes",    m_model_fp8["expert_w"],     m_hand_fp8["eW"])
    chk("  FP8 total bytes",       m_model_fp8["total"],        m_hand_fp8["total"])
    print(f"    breakdown FP8: KV {m_hand_fp8['kv']/GB:.2f} GB  "
          f"IDX {m_hand_fp8['idx']/GB:.2f} GB  nonW {m_hand_fp8['nW']/GB:.2f} GB  "
          f"expW {m_hand_fp8['eW']/GB:.2f} GB  TOTAL {m_hand_fp8['total']/GB:.2f} GB")
    print()

    for mode_lbl in ("offload", "hbm"):
        for tp in (2, 4):
            ep = 8
            print(f"  ── PP=4 TP={tp} EP={ep}  FP8  ({mode_lbl} mode) ──")
            print(f"  {'scenario':>14}  {'mem GB':>7} {'fit?':>5}  "
                  f"{'DSA step':>9} {'IC step':>9} {'speedup':>8}  "
                  f"{'DSA TPOT':>9} {'IC TPOT':>9}  "
                  f"{'DSA tok/s':>10} {'IC tok/s':>10}")
            print("  " + "-" * 110)
            for label, sl_i, bs_i in scenarios:
                m = per_rank_memory_bytes(4, tp, ep, sl_i, bs_i, fp8=True)
                fits = m["total"] / GB + 8 <= 80
                dsa_costs = build_pattern(NUM_LAYERS, 1, sl_i, bs_i, mode_lbl, ep=ep, fp8=True)
                ic_costs  = build_pattern(NUM_LAYERS, 4, sl_i, bs_i, mode_lbl, ep=ep, fp8=True)
                dsa = schedule_pp(dsa_costs, pp=4, tp=tp)
                ic  = schedule_pp(ic_costs,  pp=4, tp=tp)
                dsa_tput = bs_i * 1000.0 / dsa["total_ms"]
                ic_tput  = bs_i * 1000.0 / ic["total_ms"]
                print(f"  {label:>14}  {m['total']/GB:>6.2f}   {'YES' if fits else 'NO':>4}   "
                      f"{dsa['total_ms']:>8.3f}  {ic['total_ms']:>8.3f}  "
                      f"{dsa['total_ms']/ic['total_ms']:>7.2f}x  "
                      f"{dsa['total_ms']:>8.3f}  {ic['total_ms']:>8.3f}  "
                      f"{dsa_tput:>9.1f}  {ic_tput:>9.1f}")
            print()


if __name__ == "__main__":
    main()
