#!/usr/bin/env python3
"""DSA-native (no IndexCache) batch-size sweep at sl=128K.

Config: PP=2 TP=2 EP=8 FP8 (16 H100s), HBM-resident, all-F layers (DSA),
MLA absorption ENABLED — i.e. the kv_up_proj GEMM is removed because
W^UK is absorbed into Q and W^UV into the latent attention path before
o_proj, matching DeepSeek's reference inference.

For each batch size we report:
  - Per-rank memory breakdown (expert, dense, KV, IDX, CUDA scratch, total)
  - Whether it fits in 80 GB × 90% = 72 GB usable HBM
  - TPOT (ms / output token) under the producer/consumer pipeline schedule
  - tokens/s/rank and aggregate tokens/s across the 16-GPU cluster
"""

from __future__ import annotations

import math

from experiments.analytical_layer import (
    NUM_LAYERS,
    analytical_layer_F,
    per_rank_memory_bytes,
)

GB = 1024 ** 3

# ─── Cluster config ──────────────────────────────────────────────────
PP = 2
TP = 2
EP = 8
FP8 = True
ABSORB_MLA = True
GPUS = 16                       # PP × EP (each GPU is both EP & TP/DP rank)
SEQ_LEN = 128 * 1024
HBM_PER_GPU_GB = 80
HBM_USABLE_FRAC = 0.90          # leave 10% for fragmentation
CUDA_SCRATCH_GB = 3             # CUDA context + activation scratch (BS=1 baseline)

# Batch sizes to sweep
BATCH_SIZES = [1, 2, 4, 6, 8, 12, 16, 24]


def schedule_pp(layer_costs, pp, tp, n_layers=NUM_LAYERS):
    """Producer/consumer pipeline: stage-0 interleaves IO and compute;
    stages 1..PP-1 prefetch their full IO on their own bus, then compute
    starts at max(prev_stage_end, this_stage_total_io)."""
    layers_per_stage = math.ceil(n_layers / pp)
    stages = [layer_costs[i*layers_per_stage:(i+1)*layers_per_stage]
              for i in range(pp)]
    stages = [s for s in stages if s]
    cum_io = 0.0
    end_comp = 0.0
    for c in stages[0]:
        cum_io += c.total_io_ms()
        end_comp = max(end_comp, cum_io) + c.total_compute_ms(tp=tp)
    for stage in stages[1:]:
        stage_io  = sum(c.total_io_ms()           for c in stage)
        stage_cmp = sum(c.total_compute_ms(tp=tp) for c in stage)
        end_comp = max(end_comp, stage_io) + stage_cmp
    return end_comp


def main():
    print()
    print("=" * 92)
    print(f" DSA-native BS sweep · PP={PP} TP={TP} EP={EP} FP8 · sl={SEQ_LEN//1024}K"
          f" · MLA absorption ON")
    print("=" * 92)
    print(f"{'BS':>3}  {'Exp W':>7} {'Dense W':>8} {'KV':>7} {'IDX':>7}"
          f" {'Scratch':>8} {'Total':>7} {'Fits?':>6}"
          f"  {'TPOT':>9} {'tok/s/rank':>11} {'cluster tok/s':>14}")
    print("-" * 92)

    budget_gb = HBM_PER_GPU_GB * HBM_USABLE_FRAC

    for bs in BATCH_SIZES:
        # ─── Per-rank memory ────────────────────────────────────────
        m = per_rank_memory_bytes(pp=PP, tp=TP, ep=EP, seq_len=SEQ_LEN,
                                  bs=bs, fp8=FP8)
        expert_gb  = m["expert_w"]    / GB
        dense_gb   = m["non_expert_w"]/ GB
        kv_gb      = m["kv"]          / GB
        idx_gb     = m["idx"]         / GB
        total_gb   = expert_gb + dense_gb + kv_gb + idx_gb + CUDA_SCRATCH_GB
        fits       = total_gb <= budget_gb

        # ─── TPOT (all-F DSA pattern, HBM mode, MLA absorption ON) ──
        layers = [analytical_layer_F(SEQ_LEN, bs, "hbm",
                                     ep=EP, fp8=FP8, absorb_mla=ABSORB_MLA)
                  for _ in range(NUM_LAYERS)]
        tpot_ms = schedule_pp(layers, pp=PP, tp=TP, n_layers=NUM_LAYERS)
        tps_per_rank = bs / tpot_ms * 1000
        # Aggregate throughput: DP_attn groups × bs / TPOT.
        # Within a stage there are 8 GPUs (= EP) hosting EP=8 / TP=2 = 4
        # attention-DP groups. Each DP group serves an independent batch.
        dp_attn = EP // TP
        cluster_tps = dp_attn * tps_per_rank

        print(f"{bs:>3}  {expert_gb:>6.1f}G {dense_gb:>7.2f}G {kv_gb:>6.2f}G "
              f"{idx_gb:>6.2f}G {CUDA_SCRATCH_GB:>7.1f}G {total_gb:>6.1f}G "
              f"{'✓' if fits else '✗':>6}  {tpot_ms:>8.2f}ms "
              f"{tps_per_rank:>10.1f} {cluster_tps:>13.1f}")

    print()
    print(f"  HBM budget per GPU: {HBM_PER_GPU_GB} × {HBM_USABLE_FRAC} - "
          f"{CUDA_SCRATCH_GB} GB scratch = {budget_gb - CUDA_SCRATCH_GB:.1f} GB"
          f" available for weights + cache")
    print(f"  DP_attn (attention-parallel groups per stage) = EP/TP = {EP//TP}")
    print(f"  Cluster tok/s = DP_attn × bs / TPOT_ms × 1000")


if __name__ == "__main__":
    main()
