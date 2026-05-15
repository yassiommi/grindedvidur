#!/usr/bin/env python3
"""DSA-native (no IndexCache) batch-size sweep at sl=128K.

Cluster: PP=2 TP=2 EP=8 FP8 (16 H100s), all-F layers (DSA),
MLA absorption ENABLED — i.e. the kv_up_proj GEMM is removed because
W^UK is absorbed into Q and W^UV into the latent attention path before
o_proj, matching DeepSeek's reference inference.

Two storage regimes:
  1. **HBM**           — MLA KV cache + indexer K both resident in HBM
  2. **KV-on-SSD**     — KV cache lives on local NVMe SSD per GPU;
                         indexer K stays in HBM
                         (two SSD speeds: 7 GB/s single Gen4, 28 GB/s 4×RAID)

For each (regime, BS) we report:
  - Per-rank HBM memory budget (always the same; SSD residency frees the
    KV component of HBM use)
  - TPOT (ms / output token) under the producer/consumer pipeline schedule
  - tokens/s/rank and aggregate cluster tokens/s (DP_attn = EP/TP = 4)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from experiments.analytical_layer import (
    DSA_ATTENDED,
    GB,
    HBM_BW_GBPS,
    HBM_FLOOR_MS,
    INDEXER_K_BYTES_PER_TOKEN,
    KV_LORA_RANK,
    NUM_LAYERS,
    QK_ROPE,
    analytical_layer_F,
    per_rank_memory_bytes,
)

# ─── Cluster config ──────────────────────────────────────────────────
PP = 2
TP = 2
EP = 8
FP8 = True
ABSORB_MLA = True
SEQ_LEN = 128 * 1024
HBM_PER_GPU_GB = 80
HBM_USABLE_FRAC = 0.90
CUDA_SCRATCH_GB = 3
BATCH_SIZES = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24]
SSD_FLOOR_MS = 0.050    # ~50 us OS + driver overhead per read submission
SSD_BW_OPTIONS = [
    ("4× Gen4 RAID", 28.0),
]


# ─── IO helpers ──────────────────────────────────────────────────────
def hbm_ms(bytes_: int) -> float:
    return max(bytes_ / (HBM_BW_GBPS * GB) * 1000.0, HBM_FLOOR_MS)


def ssd_ms(bytes_: int, bw_gbps: float) -> float:
    return max(bytes_ / (bw_gbps * GB) * 1000.0, SSD_FLOOR_MS)


@dataclass
class StageCost:
    """Per-layer cost with explicit IO/compute split for the schedule."""
    cmp_ms: float
    idx_io_ms: float
    kv_io_ms: float

    @property
    def io_ms(self) -> float:
        return self.idx_io_ms + self.kv_io_ms


# ─── Per-layer cost builders ─────────────────────────────────────────
def layer_cost_hbm(sl: int, bs: int) -> StageCost:
    """Everything on HBM. F layer (DSA: every layer is F)."""
    f = analytical_layer_F(sl, bs, "hbm", ep=EP, fp8=FP8, absorb_mla=ABSORB_MLA)
    # f.idx_io_ms and f.kv_io_ms already use HBM bandwidth.
    return StageCost(cmp_ms=f.total_compute_ms(tp=TP),
                     idx_io_ms=f.idx_io_ms,
                     kv_io_ms=f.kv_io_ms)


def layer_cost_kv_on_ssd(sl: int, bs: int, ssd_bw: float) -> StageCost:
    """Indexer K on HBM, MLA KV on local NVMe SSD."""
    f = analytical_layer_F(sl, bs, "hbm", ep=EP, fp8=FP8, absorb_mla=ABSORB_MLA)
    kv_bytes_per_tok = (KV_LORA_RANK + QK_ROPE) * (1 if FP8 else 2)
    kv_bytes  = bs * DSA_ATTENDED * kv_bytes_per_tok
    idx_bytes = bs * sl * INDEXER_K_BYTES_PER_TOKEN
    return StageCost(cmp_ms=f.total_compute_ms(tp=TP),
                     idx_io_ms=hbm_ms(idx_bytes),
                     kv_io_ms=ssd_ms(kv_bytes, ssd_bw))


def schedule_pp(layers, pp: int) -> float:
    """Producer-consumer schedule across PP stages.

    Within stage 0: cum_io grows monotonically; compute_end_i =
        max(compute_end_{i-1}, cum_io_i) + cmp_i.
    Stages 1..PP-1: full IO prefetched on the stage's own bus; compute
    starts at max(prev_stage_end, this_stage_io)."""
    layers_per_stage = math.ceil(len(layers) / pp)
    stages = [layers[i*layers_per_stage:(i+1)*layers_per_stage]
              for i in range(pp)]
    stages = [s for s in stages if s]

    cum_io, end_cmp = 0.0, 0.0
    for c in stages[0]:
        cum_io += c.io_ms
        end_cmp = max(end_cmp, cum_io) + c.cmp_ms
    for stage in stages[1:]:
        stage_io  = sum(c.io_ms  for c in stage)
        stage_cmp = sum(c.cmp_ms for c in stage)
        end_cmp = max(end_cmp, stage_io) + stage_cmp
    return end_cmp


# ─── Memory-budget helper ────────────────────────────────────────────
def memory_breakdown(bs: int, kv_on_ssd: bool) -> dict:
    """Per-rank HBM footprint. With KV on SSD, the kv term moves out of
    HBM but the indexer K (always HBM) stays."""
    m = per_rank_memory_bytes(pp=PP, tp=TP, ep=EP, seq_len=SEQ_LEN,
                              bs=bs, fp8=FP8)
    parts = {
        "expert_w":     m["expert_w"]     / GB,
        "non_expert_w": m["non_expert_w"] / GB,
        "kv":           0.0 if kv_on_ssd else m["kv"] / GB,
        "idx":          m["idx"] / GB,
        "cuda_scratch": CUDA_SCRATCH_GB,
    }
    parts["total"] = sum(parts.values())
    return parts


# ─── Report runners ──────────────────────────────────────────────────
def fits(total_gb: float) -> bool:
    return total_gb <= HBM_PER_GPU_GB * HBM_USABLE_FRAC


def print_header(label: str):
    print()
    print("=" * 110)
    print(f" DSA-native BS sweep · {label} · PP={PP} TP={TP} EP={EP} FP8 · "
          f"sl={SEQ_LEN // 1024}K · MLA absorption ON")
    print("=" * 110)
    print(f"  Terminology:")
    print(f"    BS/rank      = requests served by ONE attention rank")
    print(f"    BS cluster   = total concurrent requests across all DP_attn={EP//TP} attention groups")
    print(f"                 = DP_attn × BS/rank  (independent batches in parallel)")
    print("-" * 110)
    print(f"{'BS/rank':>8} {'BS cluster':>11}  "
          f"{'Exp W':>7} {'Dense':>7} {'KV':>7} {'IDX':>7}"
          f" {'Scratch':>8} {'Total':>7} {'Fits?':>6}"
          f"  {'TPOT':>10} {'tok/s/rank':>11} {'cluster tok/s':>14}")
    print("-" * 110)


def print_row(bs: int, mem: dict, tpot_ms: float):
    tps_per_rank = bs / tpot_ms * 1000
    dp_attn = EP // TP
    cluster_tps = dp_attn * tps_per_rank
    bs_cluster = bs * dp_attn
    print(f"{bs:>8} {bs_cluster:>11}  "
          f"{mem['expert_w']:>6.1f}G {mem['non_expert_w']:>6.2f}G "
          f"{mem['kv']:>6.2f}G {mem['idx']:>6.2f}G "
          f"{mem['cuda_scratch']:>7.1f}G {mem['total']:>6.1f}G "
          f"{'✓' if fits(mem['total']) else '✗':>6}  "
          f"{tpot_ms:>9.2f}ms {tps_per_rank:>10.1f} {cluster_tps:>13.1f}")


def run_hbm():
    print_header("HBM (KV + indexer both in HBM)")
    for bs in BATCH_SIZES:
        mem = memory_breakdown(bs, kv_on_ssd=False)
        layers = [layer_cost_hbm(SEQ_LEN, bs) for _ in range(NUM_LAYERS)]
        tpot = schedule_pp(layers, pp=PP)
        print_row(bs, mem, tpot)


def run_kv_on_ssd(label: str, ssd_bw: float):
    print_header(f"KV-on-SSD ({label}, {ssd_bw} GB/s) · indexer in HBM")
    for bs in BATCH_SIZES:
        mem = memory_breakdown(bs, kv_on_ssd=True)
        layers = [layer_cost_kv_on_ssd(SEQ_LEN, bs, ssd_bw)
                  for _ in range(NUM_LAYERS)]
        tpot = schedule_pp(layers, pp=PP)
        print_row(bs, mem, tpot)


def print_side_by_side():
    print()
    print("=" * 110)
    print(" Side-by-side TPOT comparison (ms / output token, per-request latency)")
    print("=" * 110)
    head = (f"{'BS/rank':>8} {'BS cluster':>11}  {'HBM':>11}  "
            + "  ".join(f"{'SSD-' + lbl:>14}" for lbl, _ in SSD_BW_OPTIONS))
    print(head)
    print("-" * 110)
    dp_attn = EP // TP
    for bs in BATCH_SIZES:
        hbm_layers = [layer_cost_hbm(SEQ_LEN, bs) for _ in range(NUM_LAYERS)]
        hbm_tpot = schedule_pp(hbm_layers, pp=PP)
        hbm_mem  = memory_breakdown(bs, kv_on_ssd=False)
        hbm_str  = f"{hbm_tpot:>9.2f}ms" if fits(hbm_mem['total']) else "  overflow"
        row = f"{bs:>8} {bs * dp_attn:>11}  {hbm_str:>11}  "
        for _, bw in SSD_BW_OPTIONS:
            ssd_layers = [layer_cost_kv_on_ssd(SEQ_LEN, bs, bw)
                          for _ in range(NUM_LAYERS)]
            ssd_tpot = schedule_pp(ssd_layers, pp=PP)
            ssd_mem  = memory_breakdown(bs, kv_on_ssd=True)
            ssd_str  = f"{ssd_tpot:>9.2f}ms" if fits(ssd_mem['total']) else "  overflow"
            row += f"  {ssd_str:>12}  "
        print(row)


def print_memory_headroom_ssd():
    print()
    print("=" * 110)
    print(" Memory headroom with KV-on-SSD: max BS/rank at sl=128K")
    print("=" * 110)
    print("  KV no longer in HBM → only Expert W + Dense W + IDX + scratch consume HBM.")
    print("  All BS columns here are PER-RANK (multiply by DP_attn=4 for cluster BS).")
    print()
    print(f"{'BS/rank':>8} {'BS cluster':>11}  "
          f"{'Exp W':>7} {'Dense':>7} {'KV(SSD)':>9} {'IDX(HBM)':>10}"
          f" {'Scratch':>8} {'HBM tot':>9} {'Fits?':>6}"
          f"  {'SSD use/req':>13}")
    print("-" * 110)
    dp_attn = EP // TP
    for bs in BATCH_SIZES + [32, 48, 64]:
        mem = memory_breakdown(bs, kv_on_ssd=True)
        kv_gb_ssd = per_rank_memory_bytes(PP, TP, EP, SEQ_LEN, bs, fp8=FP8)["kv"] / GB
        print(f"{bs:>8} {bs * dp_attn:>11}  "
              f"{mem['expert_w']:>6.1f}G {mem['non_expert_w']:>6.2f}G "
              f"{kv_gb_ssd:>8.2f}G {mem['idx']:>9.2f}G "
              f"{mem['cuda_scratch']:>7.1f}G {mem['total']:>8.1f}G "
              f"{'✓' if fits(mem['total']) else '✗':>6}  "
              f"{kv_gb_ssd:>11.2f} GB")


def main():
    run_hbm()
    for label, bw in SSD_BW_OPTIONS:
        run_kv_on_ssd(label, bw)
    print_side_by_side()
    print_memory_headroom_ssd()
    print()
    print("  HBM budget per GPU: 80 × 0.9 − 3 (scratch) = 69 GB usable for weights + cache")
    print("  DP_attn (attention-parallel groups per stage) = EP/TP = 4")
    print()
    print("  BS/rank vs BS cluster:")
    print("    * BS/rank   — what the kernel sees: number of requests an attention rank batches together.")
    print("                  TPOT is determined by this (compute and IO scale with BS/rank).")
    print("    * BS cluster — total concurrent requests across the whole 16-GPU job:")
    print("                  = DP_attn × BS/rank.   DP groups serve independent batches in parallel.")
    print("    * Cluster tok/s = DP_attn × BS/rank / TPOT × 1000")
    print("                    = BS cluster / TPOT × 1000")
    print("    * DP multiplies concurrency, NOT latency — TPOT per request stays the same.")


if __name__ == "__main__":
    main()
