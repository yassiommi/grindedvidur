#!/usr/bin/env python3
"""DSA vs IndexCache when KV cache is offloaded to SSD but indexer K stays in HBM.

Config: PP=2, TP=2, EP=8, FP8. The MoE expert weights and indexer K live in
HBM; the MLA KV cache is read from a local NVMe SSD per decode step.

The pipeline scheduler is the same as analytical_layer / validate_pp2_tp:
- Each PP stage has its own IO bus (HBM + SSD treated as a single serialized
  per-stage IO stream for simplicity, since the GPU has to issue both reads).
- Stage 0 interleaves IO and compute (producer/consumer).
- Stage k>0 starts compute at max(prev_stage_end, this_stage_total_io_end).

SSD bandwidth modeled as a single read stream per GPU (one NVMe per GPU).
We report two assumptions: 7 GB/s (Gen4 single drive) and 28 GB/s (4× Gen4
NVMe in RAID).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from experiments.analytical_layer import (
    NUM_LAYERS,
    HBM_BW_GBPS,
    HBM_FLOOR_MS,
    INDEXER_K_BYTES_PER_TOKEN,
    KV_LORA_RANK,
    QK_ROPE,
    DSA_ATTENDED,
    GB,
    analytical_layer_F,
    analytical_layer_S,
)

SSD_BW_OPTIONS = [
    ("Gen4 NVMe single drive",   7.0),
    ("4× Gen4 NVMe RAID",       28.0),
]
SSD_FLOOR_MS = 0.050  # ~5 us OS + driver overhead per read submission


def hbm_ms(bytes_: int) -> float:
    return max(bytes_ / (HBM_BW_GBPS * GB) * 1000.0, HBM_FLOOR_MS)


def ssd_ms(bytes_: int, bw_gbps: float) -> float:
    return max(bytes_ / (bw_gbps * GB) * 1000.0, SSD_FLOOR_MS)


@dataclass
class StageCost:
    kind: str        # "F" or "S"
    cmp_ms: float    # TP-sharded compute on the GPU
    idx_io_ms: float # indexer-K HBM read (F only)
    kv_io_ms: float  # KV SSD read (both F and S)

    @property
    def io_ms(self) -> float:
        return self.idx_io_ms + self.kv_io_ms


def layer_cost(kind: str, seq_len: int, bs: int, tp: int, fp8: bool,
               ep: int, ssd_bw: float) -> StageCost:
    """Costs for one transformer layer with KV on SSD, indexer-K in HBM.

    Reuses analytical_layer_{F,S} for the compute side (which is exactly the
    same), but overrides the IO times to use SSD for KV and HBM for idx.
    """
    f = analytical_layer_F(seq_len, bs, "hbm", ep=ep, fp8=fp8)
    s = analytical_layer_S(seq_len, bs, "hbm", ep=ep, fp8=fp8)
    base = f if kind == "F" else s

    kv_bytes_per_tok = (KV_LORA_RANK + QK_ROPE) * (1 if fp8 else 2)
    idx_bytes = bs * seq_len * INDEXER_K_BYTES_PER_TOKEN if kind == "F" else 0
    kv_bytes  = bs * DSA_ATTENDED * kv_bytes_per_tok

    return StageCost(
        kind=kind,
        cmp_ms=base.total_compute_ms(tp=tp),
        idx_io_ms=hbm_ms(idx_bytes) if kind == "F" else 0.0,
        kv_io_ms=ssd_ms(kv_bytes, ssd_bw),
    )


def schedule_pp(layers, pp: int) -> float:
    """Producer-consumer schedule across PP stages.

    Within a stage: cum_io grows monotonically; compute_end_i = max(
        compute_end_{i-1}, cum_io_i) + cmp_i. Across stages: stage k starts
    compute at max(prev_stage_end, stage_k_total_io)."""
    layers_per_stage = math.ceil(len(layers) / pp)
    stages = [layers[i*layers_per_stage:(i+1)*layers_per_stage]
              for i in range(pp)]
    stages = [s for s in stages if s]

    # Stage 0: interleaved
    cum_io, end_cmp = 0.0, 0.0
    for c in stages[0]:
        cum_io += c.io_ms
        end_cmp = max(end_cmp, cum_io) + c.cmp_ms

    # Stages 1..PP-1
    for stage in stages[1:]:
        stage_io  = sum(c.io_ms  for c in stage)
        stage_cmp = sum(c.cmp_ms for c in stage)
        end_cmp = max(end_cmp, stage_io) + stage_cmp

    return end_cmp


def build_pattern(scheme: str, seq_len: int, bs: int, tp: int,
                  fp8: bool, ep: int, ssd_bw: float):
    """scheme: 'DSA' (all F) or 'IC' (F:S:S:S)."""
    f_period = 1 if scheme == "DSA" else 4
    layers = []
    for i in range(NUM_LAYERS):
        kind = "F" if i % f_period == 0 else "S"
        layers.append(layer_cost(kind, seq_len, bs, tp, fp8, ep, ssd_bw))
    return layers


SCENARIOS = [
    # (label, seq_len, bs)
    ("32K   BS=1",   32 * 1024,   1),
    ("128K  BS=1",   128 * 1024,  1),
    ("200K  BS=1",   200 * 1024,  1),
    ("512K  BS=1",   512 * 1024,  1),
    ("1M    BS=1",   1024 * 1024, 1),
    ("2M    BS=1",   2 * 1024 * 1024, 1),
    ("4M    BS=1",   4 * 1024 * 1024, 1),
    ("32K   BS=8",   32 * 1024,   8),
    ("128K  BS=8",   128 * 1024,  8),
    ("200K  BS=8",   200 * 1024,  8),
    ("512K  BS=8",   512 * 1024,  8),
    ("1M    BS=8",   1024 * 1024, 8),
    ("2M    BS=8",   2 * 1024 * 1024, 8),
    ("32K   BS=32",  32 * 1024,  32),
    ("128K  BS=32",  128 * 1024, 32),
    ("200K  BS=32",  200 * 1024, 32),
    ("512K  BS=32",  512 * 1024, 32),
]


def run_one(ssd_bw: float, label: str):
    print(f"\n{'='*94}")
    print(f"PP=2 TP=2 EP=8 FP8 · KV → SSD ({label}, {ssd_bw} GB/s) · indexer-K in HBM")
    print(f"{'='*94}")
    print(f"{'scenario':<14}{'DSA TPOT':>11}{'IC TPOT':>11}{'speedup':>10}"
          f"{'DSA tok/s':>12}{'IC tok/s':>12}")
    print("-" * 70)
    for sl_lbl, sl, bs in SCENARIOS:
        dsa = schedule_pp(build_pattern("DSA", sl, bs, 2, True, 8, ssd_bw), 2)
        ic  = schedule_pp(build_pattern("IC",  sl, bs, 2, True, 8, ssd_bw), 2)
        sp  = dsa / ic
        dsa_tps = bs / dsa * 1000
        ic_tps  = bs / ic  * 1000
        print(f"{sl_lbl:<14}{dsa:>10.2f}ms{ic:>10.2f}ms"
              f"{sp:>9.2f}×{dsa_tps:>11.1f}{ic_tps:>11.1f}")


def main():
    for label, bw in SSD_BW_OPTIONS:
        run_one(bw, label)

    # Side-by-side speedup summary
    print("\n" + "=" * 70)
    print("IndexCache speedup vs DSA — same workload, two SSD assumptions")
    print("=" * 70)
    hdr = f"{'scenario':<14}" + "".join(f"{lbl:>22}" for lbl, _ in SSD_BW_OPTIONS)
    print(hdr)
    for sl_lbl, sl, bs in SCENARIOS:
        row = f"{sl_lbl:<14}"
        for _, bw in SSD_BW_OPTIONS:
            dsa = schedule_pp(build_pattern("DSA", sl, bs, 2, True, 8, bw), 2)
            ic  = schedule_pp(build_pattern("IC",  sl, bs, 2, True, 8, bw), 2)
            row += f"{dsa/ic:>21.2f}×"
        print(row)


if __name__ == "__main__":
    main()
