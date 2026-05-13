#!/usr/bin/env python3
"""Generate figures for the IndexCache weekly slides.

Writes PNGs to reports/figures/indexcache_slides/. All numbers come
from experiments/analytical_layer.py (pure analytical, no profiled CSV).
"""

from __future__ import annotations

import math
import os
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from experiments.analytical_layer import (
    NUM_LAYERS,
    analytical_layer_F,
    analytical_layer_S,
)

OUT = "reports/figures/indexcache_slides"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Schedule (lift the producer/consumer logic from validate_pp2_tp.py)
# ─────────────────────────────────────────────────────────────────────
def schedule_pp(layer_costs, pp, tp, n_layers=NUM_LAYERS):
    layers_per_stage = math.ceil(n_layers / pp)
    stages = [layer_costs[i*layers_per_stage:(i+1)*layers_per_stage]
              for i in range(pp)]
    stages = [s for s in stages if s]
    cum_io = 0.0
    end_comp = 0.0
    for c in stages[0]:
        cum_io += c.total_io_ms()
        comp = c.total_compute_ms(tp=tp)
        end_comp = max(end_comp, cum_io) + comp
    for k in range(1, len(stages)):
        stage = stages[k]
        stage_io = sum(c.total_io_ms() for c in stage)
        stage_cmp = sum(c.total_compute_ms(tp=tp) for c in stage)
        end_comp = max(end_comp, stage_io) + stage_cmp
    return end_comp


def build_pattern(n_layers, f_period, seq_len, bs, mode, ep, fp8=False):
    return [analytical_layer_F(seq_len, bs, mode, ep, fp8=fp8)
            if i % f_period == 0
            else analytical_layer_S(seq_len, bs, mode, ep, fp8=fp8)
            for i in range(n_layers)]


# ─────────────────────────────────────────────────────────────────────
# Figure 1 — TPOT vs seq_len at BS=1 in HBM mode (FP16 and FP8)
# ─────────────────────────────────────────────────────────────────────
def fig_tpot_vs_sl():
    sls = [4*1024, 32*1024, 128*1024, 200*1024, 512*1024,
           1024*1024, 2*1024*1024, 4*1024*1024]
    sl_labels = ['4K', '32K', '128K', '200K', '512K', '1M', '2M', '4M']

    series = {}  # (dtype, scheme) -> list of TPOT
    for fp8 in (False, True):
        for f_period, scheme in [(1, "DSA"), (4, "IC (F:S:S:S)")]:
            ys = []
            for sl in sls:
                costs = build_pattern(NUM_LAYERS, f_period, sl, 1, "hbm", ep=8, fp8=fp8)
                ys.append(schedule_pp(costs, pp=4, tp=4))
            series[("FP8" if fp8 else "FP16", scheme)] = ys

    fig, ax = plt.subplots(figsize=(8, 4.8))
    color_FP16 = "#3b6bb0"
    color_FP8  = "#e08214"
    ax.plot(sls, series[("FP16", "DSA")],         "o--", color=color_FP16, label="FP16 — DSA",        linewidth=2, markersize=6)
    ax.plot(sls, series[("FP16", "IC (F:S:S:S)")],"o-",  color=color_FP16, label="FP16 — IndexCache", linewidth=2.5, markersize=6)
    ax.plot(sls, series[("FP8",  "DSA")],         "s--", color=color_FP8,  label="FP8 — DSA",         linewidth=2, markersize=6)
    ax.plot(sls, series[("FP8",  "IC (F:S:S:S)")],"s-",  color=color_FP8,  label="FP8 — IndexCache",  linewidth=2.5, markersize=6)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sls)
    ax.set_xticklabels(sl_labels)
    ax.set_xlabel("seq_len (log scale)")
    ax.set_ylabel("TPOT  (ms / output token)")
    ax.set_title("HBM-mode TPOT vs seq_len  ·  BS=1  ·  PP=4 TP=4 EP=8")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=10, frameon=True)

    # Annotate the crossovers
    ax.annotate("HBM crossover\n(FP8 ≈ 1M, FP16 ≈ 2M)",
                xy=(1.5e6, 22), xytext=(2.2e5, 33),
                fontsize=10, color="#444",
                arrowprops=dict(arrowstyle="->", color="#666", lw=1))

    plt.tight_layout()
    path = os.path.join(OUT, "fig_tpot_vs_sl_hbm_bs1.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"  wrote {path}")


# ─────────────────────────────────────────────────────────────────────
# Figure 2 — Speedup bars: HBM vs offload at key corners
# ─────────────────────────────────────────────────────────────────────
def fig_speedup_hbm_vs_offload():
    corners = [
        ("200K\nBS=1\nFP16",  200*1024,  1, False),
        ("1M\nBS=1\nFP16",    1024*1024, 1, False),
        ("200K\nBS=8\nFP16",  200*1024,  8, False),
        ("512K\nBS=32\nFP8",  512*1024, 32, True),
        ("1M\nBS=1\nFP8",     1024*1024, 1, True),
        ("4M\nBS=1\nFP8",     4*1024*1024,1, True),
    ]
    hbm_speedups, off_speedups = [], []
    labels = []
    for label, sl, bs, fp8 in corners:
        labels.append(label)
        for mode, store in (("hbm", hbm_speedups), ("offload", off_speedups)):
            dsa = schedule_pp(build_pattern(NUM_LAYERS, 1, sl, bs, mode, ep=8, fp8=fp8), pp=4, tp=4)
            ic  = schedule_pp(build_pattern(NUM_LAYERS, 4, sl, bs, mode, ep=8, fp8=fp8), pp=4, tp=4)
            store.append(dsa / ic)

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars_h = ax.bar(x - w/2, hbm_speedups, w, label="HBM mode",     color="#3b6bb0")
    bars_o = ax.bar(x + w/2, off_speedups, w, label="Offload (PCIe)", color="#bf6b3a")
    ax.axhline(1.0, color="#666", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(labels)-0.5, 1.05, "1.0× = no change", color="#666", fontsize=9, ha="right")

    for bars in (bars_h, bars_o):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.04,
                    f"{b.get_height():.2f}×", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("IndexCache speedup (DSA TPOT / IC TPOT)")
    ax.set_title("IndexCache speedup: HBM vs Offload  ·  PP=4 TP=4 EP=8")
    ax.set_ylim(0.9, max(off_speedups) * 1.12)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(OUT, "fig_speedup_hbm_vs_offload.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"  wrote {path}")


# ─────────────────────────────────────────────────────────────────────
# Figure 3 — F vs S layer cost breakdown (stacked bar)
# ─────────────────────────────────────────────────────────────────────
def fig_f_vs_s_breakdown():
    F = analytical_layer_F(200*1024, 1, "hbm", ep=8)
    S = analytical_layer_S(200*1024, 1, "hbm", ep=8)

    # Group pieces visually
    pieces = [
        ("idx_io",           F.idx_io_ms,            S.idx_io_ms,            "#c44e52"),
        ("kv_io",            F.kv_io_ms,             S.kv_io_ms,             "#e0a14a"),
        ("block_a (Q proj)", F.block_a_ms,           S.block_a_ms,           "#4c8bb0"),
        ("idx_comp",         F.idx_comp_ms,          S.idx_comp_ms,          "#a04ec4"),
        ("block_c (attn+o)", F.block_c_ms,           S.block_c_ms,           "#5a9c5e"),
        ("moe shardable",    F.moe_shardable_ms,     S.moe_shardable_ms,     "#8ac270"),
        ("expert_gemm",      F.moe_expert_gemm_ms,   S.moe_expert_gemm_ms,   "#7568b0"),
        ("ep_comms",         F.moe_ep_comms_ms,      S.moe_ep_comms_ms,      "#999999"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    cats = ["F (Full)", "S (Shared)"]
    bottoms = [0.0, 0.0]
    for name, f_val, s_val, color in pieces:
        bars = ax.bar(cats, [f_val, s_val], bottom=bottoms, color=color, label=name, edgecolor="white", linewidth=0.5)
        for i, val in enumerate([f_val, s_val]):
            if val > 0.025:
                ax.text(i, bottoms[i] + val/2, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="white" if name in ("block_c (attn+o)", "block_a (Q proj)", "expert_gemm") else "black")
        bottoms[0] += f_val
        bottoms[1] += s_val

    ax.set_ylabel("per-layer time (ms)")
    ax.set_title("F vs S layer cost breakdown  ·  BS=1 sl=200K HBM TP=1")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False)
    for i, total in enumerate(bottoms):
        ax.text(i, total + 0.015, f"Σ = {total:.3f} ms", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(bottoms) * 1.18)
    plt.tight_layout()
    path = os.path.join(OUT, "fig_f_vs_s_breakdown.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"  wrote {path}")


# ─────────────────────────────────────────────────────────────────────
# Figure 4 — Pipeline timeline schematic (PP=2, 8 layers per stage, F:S:S:S)
# ─────────────────────────────────────────────────────────────────────
def fig_pipeline_timeline():
    """Illustrate the producer/consumer pipeline across PP stages.

    Uses BS=1 sl=200K offload (so IO bars are visible) at PP=2 EP=8 TP=1.
    For visual clarity we synthesise stylised proportions, not exact times.
    """
    fig, ax = plt.subplots(figsize=(9, 4.2))

    # Each PP stage gets its own IO bar and its own COMPUTE bar.
    # Numbers are illustrative — IO blocks are wider than compute (offload mode).
    pp = 2
    n_per_stage = 8  # 8 layers per stage in this cartoon
    io_F = 1.95   # ms
    io_S = 0.05
    cmp_layer = 0.78

    pattern = ['F','S','S','S','F','S','S','S']  # one stage
    # Each stage has its OWN IO bus → both start prefetching at t=0.
    stage_rows = []
    for stage_idx in range(pp):
        t = 0.0
        ios = []
        for kind in pattern:
            dt = io_F if kind == 'F' else io_S
            ios.append((t, dt, kind))
            t += dt
        stage_rows.append(ios)

    # Compute timelines (sequential, producer/consumer within stage 0; stage 1 starts after stage 0)
    cum_io = 0.0
    end_comp = 0.0
    s0_compute = []
    for kind in pattern:
        cum_io += io_F if kind == 'F' else io_S
        cmp_start = max(end_comp, cum_io)
        s0_compute.append((cmp_start, cmp_layer, kind))
        end_comp = cmp_start + cmp_layer

    # Stage 1 compute starts at max(end_comp, stage_1_io_total)
    stage_1_io_end = sum(io_F if k == 'F' else io_S for k in pattern)
    s1_start = max(end_comp, stage_1_io_end)
    s1_compute = []
    cur = s1_start
    for kind in pattern:
        s1_compute.append((cur, cmp_layer, kind))
        cur += cmp_layer

    color_F_io = "#c44e52"
    color_S_io = "#e0a14a"
    color_cmp_F = "#3b6bb0"
    color_cmp_S = "#4c8bb0"

    rows = [
        ("Stage 0 IO",      stage_rows[0],  None,  None),
        ("Stage 0 Compute", None,           s0_compute, None),
        ("Stage 1 IO",      stage_rows[1],  None,  None),
        ("Stage 1 Compute", None,           s1_compute, None),
    ]
    y_positions = list(range(len(rows)-1, -1, -1))  # top to bottom

    for y, (label, io_blocks, cmp_blocks, _) in zip(y_positions, rows):
        if io_blocks is not None:
            for (t0, dt, kind) in io_blocks:
                ax.barh(y, dt, left=t0, color=(color_F_io if kind=='F' else color_S_io),
                        height=0.55, edgecolor="white", linewidth=0.6)
        if cmp_blocks is not None:
            for (t0, dt, kind) in cmp_blocks:
                ax.barh(y, dt, left=t0, color=(color_cmp_F if kind=='F' else color_cmp_S),
                        height=0.55, edgecolor="white", linewidth=0.6)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("time (ms)")
    ax.set_title("Producer / consumer pipeline (cartoon, PP=2, F:S:S:S, offload IO)",
                 pad=14)

    # Legend
    handles = [
        mpatches.Patch(color=color_F_io,  label="IO — F layer (idx + KV)"),
        mpatches.Patch(color=color_S_io,  label="IO — S layer (KV only)"),
        mpatches.Patch(color=color_cmp_F, label="Compute — F layer"),
        mpatches.Patch(color=color_cmp_S, label="Compute — S layer"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=9, ncol=4, frameon=False)
    ax.set_xlim(0, max(s1_compute[-1][0] + s1_compute[-1][1], 16) * 1.05)
    ax.grid(axis="x", alpha=0.2)

    # Vertical line marking stage 0 end
    ax.axvline(end_comp, color="#666", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(end_comp + 0.1, len(rows) - 0.5, "stage 0 done",
            fontsize=9, color="#444", ha="left")
    # Annotate that stage 1 IO finished before stage 1 compute starts
    stage_1_io_end_t = sum(io_F if k=='F' else io_S for k in pattern)
    ax.axvline(stage_1_io_end_t, color="#aa3333", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(stage_1_io_end_t + 0.1, 0.5, "stage 1 IO\nprefetched",
            fontsize=8, color="#aa3333", ha="left")

    plt.tight_layout()
    path = os.path.join(OUT, "fig_pipeline_timeline.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"  wrote {path}")


# ─────────────────────────────────────────────────────────────────────
# Figure 5 — F/S layer-pattern schematic (61 layers, F:S:S:S)
# ─────────────────────────────────────────────────────────────────────
def fig_fs_schematic():
    fig, ax = plt.subplots(figsize=(11, 1.5))
    n = NUM_LAYERS
    for i in range(n):
        kind = "F" if i % 4 == 0 else "S"
        color = "#c44e52" if kind == "F" else "#9ebadb"
        ax.barh(0, 1, left=i, color=color, edgecolor="white", linewidth=1)
        if i % 4 == 0:
            ax.text(i + 0.5, 0, "F", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    ax.set_xlim(0, n)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 15, 30, 45, 60])
    ax.set_xticklabels(["layer 0", "15", "30", "45", "60"])
    ax.set_title("DeepSeek-V3 61-layer stack with F:S:S:S IndexCache pattern  ·  "
                 "16 F + 45 S layers")
    handles = [
        mpatches.Patch(color="#c44e52", label="F (Full): run indexer, materialise top-k"),
        mpatches.Patch(color="#9ebadb", label="S (Shared): reuse last F layer's top-k"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    path = os.path.join(OUT, "fig_fs_schematic.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path}")


if __name__ == "__main__":
    print(f"Writing slide figures to {OUT}/")
    fig_fs_schematic()
    fig_f_vs_s_breakdown()
    fig_pipeline_timeline()
    fig_tpot_vs_sl()
    fig_speedup_hbm_vs_offload()
