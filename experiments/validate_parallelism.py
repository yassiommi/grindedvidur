#!/usr/bin/env python3
"""Validate the parallelism-aware DSA + IndexCache model end-to-end.

Goal: for the chosen config (PP=8, TP=1, EP=8 → 64 H100s), at
seq_len=200K BS=1 in offload mode, derive every number by hand and
check the model's output against it.

Run:
    python -m experiments.validate_parallelism
"""

from __future__ import annotations

import math

from experiments.dsa_timing_model import (
    DSA_ATTENDED,
    DSA_SELECTED_TOKENS,
    DSA_SLIDING_WINDOW,
    EXPERT_WEIGHT_BYTES,
    H100_HBM_PEAK_GBS,
    H100_PCIE_PEAK_GBS,
    HIDDEN_SIZE,
    INDEXER_K_BYTES_PER_TOKEN,
    KV_LORA_RANK,
    MLA_KV_BYTES_PER_TOKEN,
    NUM_HEADS,
    NUM_LAYERS,
    NUM_ROUTED_EXPERTS,
    Q_LORA_RANK,
    load_profiles,
)
from experiments.indexcache_model import (
    EXPERT_W_PER_LAYER_B,
    NON_EXPERT_W_PER_LAYER_B,
    ParallelismConfig,
    build_layer_costs,
    fits_in_hbm,
    per_rank_memory_bytes,
    schedule_pipelined_pp,
    simulate_pp,
)

GB = 1024 ** 3
MB = 1024 ** 2


def hr(label: str) -> None:
    print()
    print("=" * 92)
    print(f" {label}")
    print("=" * 92)


def chk(name: str, got, want, tol: float = 1e-3):
    """Compare model number to hand-derived number."""
    if abs(got - want) <= tol * max(abs(got), abs(want), 1.0):
        print(f"  [OK ]  {name:<52}  model={got:>12.6f}  hand={want:>12.6f}")
    else:
        print(f"  [FAIL] {name:<52}  model={got:>12.6f}  hand={want:>12.6f}  diff={got - want:.6e}")


def main() -> None:
    print("Parallelism-aware DSA + IndexCache validation")
    print(f"H100 SXM × 64 (8 nodes × 8 GPUs), DeepSeek-V3 ({NUM_LAYERS} layers)")
    print(f"Chosen config: PP=8, TP=1, EP=8 (no MLA duplication tax)")
    print(f"Workload: seq_len=200K, BS=1, offload mode")

    # ===== STEP 1: Hand-derive everything =====
    hr("STEP 1 — Hand-derived constants and per-rank state")

    cfg = ParallelismConfig(pp=8, tp=1, ep=8)
    sl  = 200 * 1024   # 204800 tokens
    bs  = 1
    mode = "offload"

    # Weight constants (hand-derived from architecture):
    # non-expert per layer: q_down + q_up + kv_down + kv_up + o_proj + indexer_K_proj
    #                       + router_gate + shared_expert (all FP16)
    nonexp_hand = (
        HIDDEN_SIZE * Q_LORA_RANK                                  # q_down  : 7168 * 1536
        + Q_LORA_RANK * NUM_HEADS * (128 + 64)                     # q_up    : 1536 * 128 * 192
        + HIDDEN_SIZE * KV_LORA_RANK                               # kv_down : 7168 * 512
        + KV_LORA_RANK * NUM_HEADS * (128 + 128)                   # kv_up   : 512 * 128 * 256
        + NUM_HEADS * 128 * HIDDEN_SIZE                            # o_proj  : 128 * 128 * 7168
        + HIDDEN_SIZE * KV_LORA_RANK                               # idx K   : 7168 * 512
        + HIDDEN_SIZE * 256                                        # router  : 7168 * 256
        + 3 * HIDDEN_SIZE * 2048                                   # shared  : 3*7168*2048
    ) * 2  # FP16
    print(f"  Non-expert weights per layer (hand): {nonexp_hand / MB:.2f} MB")
    print(f"  Non-expert weights per layer (mdl):  {NON_EXPERT_W_PER_LAYER_B / MB:.2f} MB")
    chk("non-expert weights / layer (B)", NON_EXPERT_W_PER_LAYER_B, nonexp_hand, tol=1e-9)

    exp_hand = NUM_ROUTED_EXPERTS * 3 * HIDDEN_SIZE * 2048 * 2     # 256 × 3 × 7168 × 2048 × 2B
    print(f"  Expert weights per layer    (hand): {exp_hand / GB:.2f} GB")
    chk("expert weights / layer (B)", EXPERT_W_PER_LAYER_B, exp_hand, tol=1e-9)

    # Per-rank memory under PP=8 TP=1 EP=8:
    layers_per_stage = math.ceil(NUM_LAYERS / cfg.pp)  # 8
    kv_hand  = layers_per_stage * bs * sl * MLA_KV_BYTES_PER_TOKEN
    idx_hand = layers_per_stage * bs * sl * INDEXER_K_BYTES_PER_TOKEN
    nW_hand  = layers_per_stage * NON_EXPERT_W_PER_LAYER_B // cfg.tp
    eW_hand  = layers_per_stage * EXPERT_W_PER_LAYER_B // cfg.ep
    total_hand = kv_hand + idx_hand + nW_hand + eW_hand

    mem = per_rank_memory_bytes(cfg, sl, bs)
    print(f"\n  layers_per_stage = ceil(61/8) = {layers_per_stage}")
    chk("layers_per_stage",            cfg.layers_per_stage, layers_per_stage, tol=1e-9)
    chk("per-rank KV bytes",           mem["kv"],            kv_hand,           tol=1e-9)
    chk("per-rank IDX bytes",          mem["idx"],           idx_hand,          tol=1e-9)
    chk("per-rank non-expert W bytes", mem["non_expert_w"],  nW_hand,           tol=1e-9)
    chk("per-rank expert W bytes",     mem["expert_w"],      eW_hand,           tol=1e-9)
    chk("per-rank total bytes",        mem["total"],         total_hand,        tol=1e-9)

    print(f"\n  Per-rank breakdown @ PP=8 TP=1 EP=8, BS=1, sl=200K:")
    print(f"    KV cache               = {kv_hand / GB:>7.3f} GB")
    print(f"    Indexer K cache        = {idx_hand / GB:>7.3f} GB")
    print(f"    Non-expert weights     = {nW_hand / GB:>7.3f} GB  (/ TP=1 / PP=8)")
    print(f"    Expert weights         = {eW_hand / GB:>7.3f} GB  (/ EP=8 / PP=8)")
    print(f"    Total per rank         = {total_hand / GB:>7.3f} GB")
    print(f"    Fits in 80 GB w/ 8 GB headroom? {'YES' if fits_in_hbm(cfg, sl, bs) else 'no'}")

    # ===== STEP 2: Per-layer cost (single-rank reference) =====
    hr("STEP 2 — Per-layer cost (single-rank reference, used as-is for TP=1)")

    tables = load_profiles()
    # DSA all-F
    costs_dsa = build_layer_costs(sl, bs, mode, f_period=1, tables=tables)
    # IndexCache F:S:S:S
    costs_ic  = build_layer_costs(sl, bs, mode, f_period=4, tables=tables)

    # Hand-derive layer 0 (F layer) timing in offload mode at sl=200K BS=1:
    # idx_io  = (1 × 200K × 512 B) / (51.5 GB/s × 1024^3) × 1000
    pcie_Bps = H100_PCIE_PEAK_GBS * GB
    idx_io_hand = (bs * sl * INDEXER_K_BYTES_PER_TOKEN) / pcie_Bps * 1000
    kv_io_hand  = (bs * DSA_ATTENDED * MLA_KV_BYTES_PER_TOKEN) / pcie_Bps * 1000

    print(f"  Hand-derived (offload BS=1 sl=200K):")
    print(f"    indexer K size  = {bs * sl * INDEXER_K_BYTES_PER_TOKEN / MB:.2f} MB")
    print(f"    indexer K read  = {idx_io_hand:.4f} ms  (size / 51.5 GB/s)")
    print(f"    KV gather size  = {bs * DSA_ATTENDED * MLA_KV_BYTES_PER_TOKEN / MB:.2f} MB")
    print(f"    KV gather read  = {kv_io_hand:.4f} ms")

    print(f"\n  Model F-layer (DSA L0):")
    print(f"    idx_io  = {costs_dsa[0].idx_io_ms:.4f} ms")
    print(f"    kv_io   = {costs_dsa[0].kv_io_ms:.4f} ms")
    print(f"    block_a = {costs_dsa[0].block_a_ms:.4f} ms")
    print(f"    block_c = {costs_dsa[0].block_c_ms:.4f} ms")
    print(f"    moe     = {costs_dsa[0].moe_ms:.4f} ms")
    print(f"    T_compute_layer = {costs_dsa[0].total_compute:.4f} ms")
    chk("F-layer idx_io_ms (offload, 100 MB / PCIe)", costs_dsa[0].idx_io_ms, idx_io_hand, tol=5e-3)
    chk("F-layer kv_io_ms  (offload, 2.81 MB / PCIe)", costs_dsa[0].kv_io_ms, kv_io_hand, tol=5e-3)

    print(f"\n  Model S-layer (IC L1):  idx_io=0  kv_io={costs_ic[1].kv_io_ms:.4f} ms  "
          f"T_compute={costs_ic[1].total_compute:.4f}")
    chk("S-layer idx_io_ms must be 0", costs_ic[1].idx_io_ms, 0.0, tol=1e-9)

    # ===== STEP 3: Aggregate IO and compute =====
    hr("STEP 3 — Aggregate IO and compute across 61 layers")

    total_io_dsa  = sum(c.total_io for c in costs_dsa)
    total_io_ic   = sum(c.total_io for c in costs_ic)
    total_cmp_dsa = sum(c.total_compute for c in costs_dsa)
    total_cmp_ic  = sum(c.total_compute for c in costs_ic)

    # Hand:
    nF_dsa = 61
    nF_ic  = math.ceil(61 / 4)  # 16
    io_dsa_hand = nF_dsa * idx_io_hand + 61 * kv_io_hand
    io_ic_hand  = nF_ic  * idx_io_hand + 61 * kv_io_hand
    print(f"  DSA: nF=61, total IO = 61 × idx_io + 61 × kv_io")
    print(f"       hand = 61 × {idx_io_hand:.4f} + 61 × {kv_io_hand:.4f} = {io_dsa_hand:.3f} ms")
    print(f"       model = {total_io_dsa:.3f} ms")
    chk("DSA total IO ms",  total_io_dsa, io_dsa_hand, tol=5e-3)

    print(f"  IC:  nF=16, total IO = 16 × idx_io + 61 × kv_io")
    print(f"       hand = 16 × {idx_io_hand:.4f} + 61 × {kv_io_hand:.4f} = {io_ic_hand:.3f} ms")
    print(f"       model = {total_io_ic:.3f} ms")
    chk("IC total IO ms",  total_io_ic,  io_ic_hand,  tol=5e-3)

    print(f"\n  DSA total compute: {total_cmp_dsa:.3f} ms (61 × T_F_compute)")
    print(f"  IC  total compute: {total_cmp_ic:.3f} ms  (16 × T_F + 45 × T_S)")
    # Hand for IC compute: T_F includes idx_comp (0.01ms), T_S doesn't
    cmp_F = costs_dsa[0].total_compute
    cmp_S = costs_ic[1].total_compute
    cmp_ic_hand = 16 * cmp_F + 45 * cmp_S
    print(f"    16 × {cmp_F:.4f} + 45 × {cmp_S:.4f} = {cmp_ic_hand:.3f} ms")
    chk("IC total compute by composition", total_cmp_ic, cmp_ic_hand, tol=1e-9)

    # ===== STEP 4: PP=8 step time, hand-derived per stage =====
    hr("STEP 4 — Step time under PP=8 (per-stage hand derivation)")

    sched_dsa = schedule_pipelined_pp(costs_dsa, cfg)
    sched_ic  = schedule_pipelined_pp(costs_ic,  cfg)

    # Split layers into stages. layers_per_stage = ceil(61/8) = 8.
    # Stages 0..6 have 8 layers each; stage 7 has 5 layers (= 61 - 7×8).
    lps = layers_per_stage
    stages_dsa = [costs_dsa[i*lps:(i+1)*lps] for i in range(cfg.pp) if costs_dsa[i*lps:(i+1)*lps]]
    stages_ic  = [costs_ic[i*lps:(i+1)*lps]  for i in range(cfg.pp) if costs_ic[i*lps:(i+1)*lps]]
    print(f"  Stages: {[len(s) for s in stages_dsa]} layers each (sum = {sum(len(s) for s in stages_dsa)})")

    def hand_pp_schedule(stages, label: str) -> float:
        """Reproduce the per-stage scheduler by hand."""
        # Stage 0: producer/consumer through its layers
        from experiments.indexcache_model import schedule_pipelined as _pc
        s0 = _pc(stages[0])
        end_compute = s0["total_ms"]
        s0_io = sum(c.total_io for c in stages[0])
        s0_cmp = sum(c.total_compute for c in stages[0])
        print(f"    {label} stage 0 ({len(stages[0])} layers): io_total={s0_io:.3f}, "
              f"compute_total={s0_cmp:.3f}, producer/consumer pipeline end = {end_compute:.3f}")
        for k in range(1, len(stages)):
            stage_io = sum(c.total_io for c in stages[k])
            stage_cmp = sum(c.total_compute for c in stages[k])
            compute_start = max(end_compute, stage_io)
            end_compute = compute_start + stage_cmp
            print(f"    {label} stage {k} ({len(stages[k])} layers): "
                  f"io={stage_io:.3f}, cmp={stage_cmp:.3f}, "
                  f"compute_start=max(prev={end_compute - stage_cmp:.3f}, io={stage_io:.3f}), "
                  f"end={end_compute:.3f}")
        return end_compute

    step_dsa_hand = hand_pp_schedule(stages_dsa, "DSA")
    print()
    step_ic_hand  = hand_pp_schedule(stages_ic,  "IC ")

    chk("DSA step_ms (per-stage hand vs model)",  sched_dsa["total_ms"], step_dsa_hand, tol=1e-9)
    chk("IC  step_ms (per-stage hand vs model)",  sched_ic["total_ms"],  step_ic_hand,  tol=1e-9)

    # ===== STEP 5: simulate_pp end-to-end =====
    hr("STEP 5 — simulate_pp end-to-end")

    s_dsa = simulate_pp(sl, bs, mode, f_period=1, cfg=cfg)
    s_ic  = simulate_pp(sl, bs, mode, f_period=4, cfg=cfg)

    chk("simulate_pp DSA step_ms", s_dsa["dsa_step_ms"], step_dsa_hand, tol=1e-9)
    chk("simulate_pp IC  step_ms", s_ic["ic_step_ms"],   step_ic_hand,  tol=1e-9)

    speedup_hand = step_dsa_hand / step_ic_hand
    print(f"\n  Speedup (DSA / IC) = {step_dsa_hand:.3f} / {step_ic_hand:.3f} = {speedup_hand:.3f}x")
    chk("speedup IC vs DSA", s_ic["speedup"], speedup_hand, tol=1e-9)

    # ===== STEP 6: Compare across PP =====
    hr("STEP 6 — Step time vs PP (BS=1, sl=200K, offload)")
    print(f"  {'PP':>3} {'feasible':>9} {'IO_wall_DSA':>11} {'IO_wall_IC':>11} "
          f"{'compute':>9} {'DSA step':>9} {'IC step':>9} {'speedup':>8} {'IC bottleneck':>14}")
    print("  " + "-" * 92)
    for pp in [1, 2, 4, 8, 16, 32]:
        try:
            c = ParallelismConfig(pp=pp, tp=1, ep=8)
        except Exception:
            continue
        s_dsa = simulate_pp(sl, bs, mode, 1, c)
        s_ic  = simulate_pp(sl, bs, mode, 4, c)
        print(f"  {pp:>3} {('YES' if s_ic['feasible'] else 'no'):>9} "
              f"{s_dsa['dsa_io_wall_ms']:>10.2f}  "
              f"{s_ic['ic_io_wall_ms']:>10.2f}  "
              f"{s_ic['ic_compute_ms']:>8.2f}  "
              f"{s_dsa['dsa_step_ms']:>8.2f}  "
              f"{s_ic['ic_step_ms']:>8.2f}  "
              f"{s_ic['speedup']:>7.2f}x  {s_ic['ic_bottleneck']:>14}")

    # ===== STEP 7: HBM mode sanity =====
    hr("STEP 7 — Same scenario, HBM mode")
    s_dsa_hbm = simulate_pp(sl, bs, "hbm", 1, cfg)
    s_ic_hbm  = simulate_pp(sl, bs, "hbm", 4, cfg)
    print(f"  PP=8 TP=1 EP=8, BS=1 sl=200K, HBM mode:")
    print(f"    DSA step: {s_dsa_hbm['dsa_step_ms']:.3f} ms  "
          f"(io_wall={s_dsa_hbm['dsa_io_wall_ms']:.3f}, cmp={s_dsa_hbm['dsa_compute_ms']:.3f})")
    print(f"    IC  step: {s_ic_hbm['ic_step_ms']:.3f} ms  "
          f"(io_wall={s_ic_hbm['ic_io_wall_ms']:.3f}, cmp={s_ic_hbm['ic_compute_ms']:.3f})")
    print(f"    Speedup: {s_ic_hbm['speedup']:.3f}x")

    # ===== STEP 8: Broader e2e at feasible corners =====
    hr("STEP 8 — End-to-end DSA vs IndexCache, PP=8 TP=1 EP=8, feasible corners only")
    cfg = ParallelismConfig(pp=8, tp=1, ep=8)
    scenarios = [
        (4 * 1024,        1),
        (32 * 1024,       1),
        (128 * 1024,      1),
        (200 * 1024,      1),
        (512 * 1024,      1),
        (1024 * 1024,     1),
        (4 * 1024,        8),
        (32 * 1024,       8),
        (128 * 1024,      8),
        (4 * 1024,       64),
        (32 * 1024,      64),
    ]
    for mode_lbl in ("offload", "hbm"):
        print(f"\n  ── {mode_lbl} ──")
        print(f"  {'sl':>5} {'bs':>3} {'mem GB':>8} {'fits':>5} "
              f"{'DSA step':>9} {'IC step':>9} {'speedup':>8} "
              f"{'IC IO wall':>11} {'IC comp':>8} {'IC bot':>8}")
        print("  " + "-" * 86)
        for sl_i, bs_i in scenarios:
            s_dsa = simulate_pp(sl_i, bs_i, mode_lbl, 1, cfg)
            s_ic  = simulate_pp(sl_i, bs_i, mode_lbl, 4, cfg)
            fits = "YES" if s_ic["feasible"] else "NO"
            sl_lab = f"{sl_i//1024}K" if sl_i < 1024*1024 else "1M"
            print(f"  {sl_lab:>5} {bs_i:>3} {s_ic['per_rank_total_gb']:>7.2f}  {fits:>4} "
                  f"{s_dsa['dsa_step_ms']:>8.2f}  "
                  f"{s_ic['ic_step_ms']:>8.2f}  "
                  f"{s_ic['speedup']:>7.2f}x "
                  f"{s_ic['ic_io_wall_ms']:>10.2f}  "
                  f"{s_ic['ic_compute_ms']:>7.2f}  "
                  f"{s_ic['ic_bottleneck']:>7}")


if __name__ == "__main__":
    main()
