#!/usr/bin/env python3
"""TPOT≤30ms feasibility heatmaps for PP=2 TP=2 EP=8 FP8 sl×BS grids.

For each (seq_len, cluster_BS) cell, compute TPOT and HBM memory fit
under four combinations:
  - DSA / HBM        (KV + indexer both in HBM)
  - IC  / HBM
  - DSA / KV-on-SSD  (KV on 28 GB/s NVMe RAID, indexer in HBM)
  - IC  / KV-on-SSD

Cell color encodes feasibility:
  - GREEN  : feasible (TPOT ≤ 30 ms AND fits in HBM)
  - YELLOW : fits in HBM but TPOT > 30 ms (compute/IO-bound)
  - RED    : doesn't fit in HBM
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from experiments.analytical_layer import (
    DSA_ATTENDED,
    GB,
    HBM_BW_GBPS,
    HBM_FLOOR_MS,
    INDEXER_DIM,
    INDEXER_K_BYTES_PER_TOKEN,
    KV_LORA_RANK,
    NUM_LAYERS,
    QK_ROPE,
    analytical_layer_F,
    analytical_layer_S,
    per_rank_memory_bytes,
)

# ─── Cluster + budget ────────────────────────────────────────────────
PP, TP, EP = 2, 2, 8
FP8 = True
ABSORB_MLA = True
DP_ATTN = EP // TP  # 4

# 80 GB × 0.9 usable − 3 GB CUDA scratch = 69 GB for weights + cache
HBM_BUDGET_GB = 80 * 0.9 - 3

# SSD spec (4× Gen4 RAID)
SSD_BW = 28.0
SSD_FLOOR_MS = 0.050

# Target acceptable TPOT
TPOT_LIMIT_MS = 30.0

KV_UNIT = KV_LORA_RANK + QK_ROPE  # 576 B FP8


# ─── Helpers ─────────────────────────────────────────────────────────
def hbm_ms(b: int) -> float:
    return max(b / (HBM_BW_GBPS * GB) * 1000.0, HBM_FLOOR_MS)


def ssd_ms(b: int) -> float:
    return max(b / (SSD_BW * GB) * 1000.0, SSD_FLOOR_MS)


def schedule_pp(layers, pp: int) -> float:
    lpst = math.ceil(len(layers) / pp)
    stages = [layers[i * lpst:(i + 1) * lpst] for i in range(pp)]
    stages = [s for s in stages if s]
    cum_io, end_cmp = 0.0, 0.0
    for c in stages[0]:
        cum_io += c["io"]
        end_cmp = max(end_cmp, cum_io) + c["cmp"]
    for stage in stages[1:]:
        sio = sum(c["io"] for c in stage)
        scmp = sum(c["cmp"] for c in stage)
        end_cmp = max(end_cmp, sio) + scmp
    return end_cmp


def tpot_ms(seq_len: int, bs_cluster: int, regime: str, scheme: str) -> float:
    bs = bs_cluster // DP_ATTN
    if bs < 1:
        return float("inf")
    f = analytical_layer_F(seq_len, bs, "hbm", ep=EP, fp8=FP8, absorb_mla=ABSORB_MLA)
    s = analytical_layer_S(seq_len, bs, "hbm", ep=EP, fp8=FP8, absorb_mla=ABSORB_MLA)
    kv_b = bs * DSA_ATTENDED * KV_UNIT
    idx_b = bs * seq_len * INDEXER_K_BYTES_PER_TOKEN

    kv_io_t = ssd_ms(kv_b) if regime == "ssd" else hbm_ms(kv_b)
    idx_io_t = hbm_ms(idx_b)

    layers = []
    for i in range(NUM_LAYERS):
        is_f_layer = (scheme == "dsa") or (i % 4 == 0)
        if is_f_layer:
            layers.append({"io": idx_io_t + kv_io_t, "cmp": f.total_compute_ms(tp=TP)})
        else:
            layers.append({"io": kv_io_t, "cmp": s.total_compute_ms(tp=TP)})
    return schedule_pp(layers, PP)


def fits_hbm(seq_len: int, bs_cluster: int, regime: str) -> bool:
    bs = bs_cluster // DP_ATTN
    if bs < 1:
        return False
    m = per_rank_memory_bytes(pp=PP, tp=TP, ep=EP, seq_len=seq_len, bs=bs, fp8=FP8)
    kv = m["kv"] / GB
    idx = m["idx"] / GB
    nw = m["non_expert_w"] / GB
    ew = m["expert_w"] / GB
    if regime == "ssd":
        return (idx + nw + ew) <= HBM_BUDGET_GB
    return (kv + idx + nw + ew) <= HBM_BUDGET_GB


# ─── Grids ──────────────────────────────────────────────────────────
SEQ_LENS = [4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024,
            128 * 1024, 200 * 1024, 256 * 1024, 512 * 1024,
            1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024]
SL_LABELS = ["4K", "8K", "16K", "32K", "64K", "128K", "200K", "256K",
             "512K", "1M", "2M", "4M"]

BS_CLUSTER = [4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384]
BS_LABELS = [str(b) for b in BS_CLUSTER]


# Status codes: 0=fits+fast (green), 1=fits+slow (yellow), 2=OOM (red)
STATUS_GREEN = 0
STATUS_YELLOW = 1
STATUS_RED = 2


def build_grid(regime: str, scheme: str):
    """Return (tpot_arr, status_arr) of shape (len(SEQ_LENS), len(BS_CLUSTER))."""
    tpot_arr = np.zeros((len(SEQ_LENS), len(BS_CLUSTER)))
    status_arr = np.zeros_like(tpot_arr, dtype=int)
    for i, sl in enumerate(SEQ_LENS):
        for j, bs in enumerate(BS_CLUSTER):
            if not fits_hbm(sl, bs, regime):
                status_arr[i, j] = STATUS_RED
                tpot_arr[i, j] = np.nan
                continue
            t = tpot_ms(sl, bs, regime, scheme)
            tpot_arr[i, j] = t
            status_arr[i, j] = STATUS_GREEN if t <= TPOT_LIMIT_MS else STATUS_YELLOW
    return tpot_arr, status_arr


# ─── Plotting ───────────────────────────────────────────────────────
def plot_panel(ax, regime: str, scheme: str, title: str):
    tpot_arr, status_arr = build_grid(regime, scheme)

    # 3-color discrete map
    cmap = mcolors.ListedColormap(["#3FAA5B", "#F0C03A", "#C84B4B"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(status_arr, cmap=cmap, norm=norm, aspect="auto", origin="lower")

    # Annotate each cell
    for i in range(len(SEQ_LENS)):
        for j in range(len(BS_CLUSTER)):
            st = status_arr[i, j]
            if st == STATUS_RED:
                txt = "OOM"
                color = "white"
            else:
                t = tpot_arr[i, j]
                if t < 100:
                    txt = f"{t:.0f}"
                else:
                    txt = f"{t:.0f}"
                color = "white" if st == STATUS_GREEN else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=7)

    ax.set_xticks(range(len(BS_CLUSTER)))
    ax.set_xticklabels(BS_LABELS, fontsize=8)
    ax.set_yticks(range(len(SEQ_LENS)))
    ax.set_yticklabels(SL_LABELS, fontsize=8)
    ax.set_xlabel("Cluster batch size", fontsize=9)
    ax.set_ylabel("Sequence length", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", which="both", bottom=True, top=False)
    ax.tick_params(axis="y", which="both", left=True, right=False)


def make_grid_figure(out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    plot_panel(axes[0, 0], "hbm", "dsa", "DSA · HBM (KV + indexer in HBM)")
    plot_panel(axes[0, 1], "hbm", "ic",  "IndexCache · HBM")
    plot_panel(axes[1, 0], "ssd", "dsa", "DSA · KV-on-SSD (28 GB/s)")
    plot_panel(axes[1, 1], "ssd", "ic",  "IndexCache · KV-on-SSD (28 GB/s)")

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#3FAA5B", label=f"feasible (TPOT ≤ {TPOT_LIMIT_MS:.0f} ms)"),
        plt.Rectangle((0, 0), 1, 1, fc="#F0C03A", label=f"fits HBM, TPOT > {TPOT_LIMIT_MS:.0f} ms"),
        plt.Rectangle((0, 0), 1, 1, fc="#C84B4B", label="OOM (does not fit HBM)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.01), frameon=False)

    fig.suptitle(
        f"Acceptable (seq_len × BS) region   ·   PP={PP} TP={TP} EP={EP} FP8   ·   "
        f"INDEXER_DIM={INDEXER_DIM}   ·   TPOT target ≤ {TPOT_LIMIT_MS:.0f} ms   ·   "
        f"HBM budget {HBM_BUDGET_GB:.0f} GB / rank",
        fontsize=11, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_single_panel(out_path: Path, regime: str, scheme: str, title: str):
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    plot_panel(ax, regime, scheme, title)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#3FAA5B", label=f"feasible (TPOT ≤ {TPOT_LIMIT_MS:.0f} ms)"),
        plt.Rectangle((0, 0), 1, 1, fc="#F0C03A", label=f"fits HBM, TPOT > {TPOT_LIMIT_MS:.0f} ms"),
        plt.Rectangle((0, 0), 1, 1, fc="#C84B4B", label="OOM (does not fit HBM)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = Path("reports/figures/feasibility")
    out_dir.mkdir(parents=True, exist_ok=True)

    make_grid_figure(out_dir / "feasibility_2x2.png")
    make_single_panel(out_dir / "feasibility_dsa_hbm.png", "hbm", "dsa",
                      "DSA · HBM (KV + indexer in HBM)")
    make_single_panel(out_dir / "feasibility_ic_hbm.png", "hbm", "ic",
                      "IndexCache · HBM")
    make_single_panel(out_dir / "feasibility_dsa_ssd.png", "ssd", "dsa",
                      "DSA · KV-on-SSD (28 GB/s)")
    make_single_panel(out_dir / "feasibility_ic_ssd.png", "ssd", "ic",
                      "IndexCache · KV-on-SSD (28 GB/s)")

    print(f"Wrote 5 figures to {out_dir}/")
    print("  feasibility_2x2.png         — all four panels")
    print("  feasibility_dsa_hbm.png     — DSA + HBM")
    print("  feasibility_ic_hbm.png      — IC  + HBM")
    print("  feasibility_dsa_ssd.png     — DSA + KV-on-SSD")
    print("  feasibility_ic_ssd.png      — IC  + KV-on-SSD")


if __name__ == "__main__":
    main()
