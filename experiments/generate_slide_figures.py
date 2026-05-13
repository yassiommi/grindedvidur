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
# Figure 4 — Pipeline Gantt: DSA vs IndexCache in HBM mode
# ─────────────────────────────────────────────────────────────────────
def fig_pipeline_timeline():
    """HBM-mode pipeline Gantt: DSA all-F vs IndexCache F:S:S:S side by side.

    Real numbers from analytical_layer at BS=1 sl=4M FP16 HBM (a regime
    where the indexer-K HBM read is comparable to per-layer compute,
    so the bars are visually informative). Cartoon PP=2 with 8 layers
    per stage so the F:S:S:S structure draws cleanly.
    """
    f = analytical_layer_F(4 * 1024 * 1024, 1, "hbm", ep=8, fp8=False)
    s = analytical_layer_S(4 * 1024 * 1024, 1, "hbm", ep=8, fp8=False)
    F_io  = f.total_io_ms()
    S_io  = s.total_io_ms()
    F_cmp = f.total_compute_ms(tp=1)
    S_cmp = s.total_compute_ms(tp=1)

    dsa_pattern = ["F"] * 8
    ic_pattern  = ["F", "S", "S", "S", "F", "S", "S", "S"]

    def simulate_pp2(pat):
        """Two stages, each with its own IO bus (independent prefetch).
        Stage 0 interleaves its own IO and compute. Stage 1 starts
        compute at max(stage 0 compute end, stage 1 IO end)."""
        # Stage 0
        cum_io, end_cmp = 0.0, 0.0
        s0_io, s0_cmp = [], []
        for kind in pat:
            dt_io  = F_io  if kind == "F" else S_io
            dt_cmp = F_cmp if kind == "F" else S_cmp
            s0_io.append((cum_io, dt_io, kind))
            cum_io += dt_io
            cmp_start = max(end_cmp, cum_io)
            s0_cmp.append((cmp_start, dt_cmp, kind))
            end_cmp = cmp_start + dt_cmp
        s0_end = end_cmp
        # Stage 1
        s1_io = []
        cum_io = 0.0
        for kind in pat:
            dt_io = F_io if kind == "F" else S_io
            s1_io.append((cum_io, dt_io, kind))
            cum_io += dt_io
        s1_io_end = cum_io
        cmp_t = max(s0_end, s1_io_end)
        s1_cmp = []
        for kind in pat:
            dt_cmp = F_cmp if kind == "F" else S_cmp
            s1_cmp.append((cmp_t, dt_cmp, kind))
            cmp_t += dt_cmp
        return s0_io, s0_cmp, s1_io, s1_cmp, cmp_t, s0_end, s1_io_end

    dsa = simulate_pp2(dsa_pattern)
    ic  = simulate_pp2(ic_pattern)
    dsa_total, ic_total = dsa[4], ic[4]
    speedup = dsa_total / ic_total

    cF_io  = "#c44e52"
    cS_io  = "#e8b554"
    cF_cmp = "#2a4a8b"
    cS_cmp = "#7aa3d0"

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                              gridspec_kw={"hspace": 0.55})
    titles = [
        f"DSA  ·  all 16 layers are F  ·  total = {dsa_total:.2f} ms",
        f"IndexCache F:S:S:S  ·  4 F + 12 S  ·  total = {ic_total:.2f} ms",
    ]
    max_t = max(dsa_total, ic_total) * 1.10
    lane_labels = ["GPU 0 · IO bus", "GPU 0 · Compute",
                   "GPU 1 · IO bus", "GPU 1 · Compute"]
    y_pos = [3, 2, 1, 0]

    for ax, run, title in zip(axes, [dsa, ic], titles):
        s0_io, s0_cmp, s1_io, s1_cmp, total, s0_end, s1_io_end = run
        for (t0, dt, kind) in s0_io:
            ax.barh(y_pos[0], dt, left=t0, color=(cF_io if kind=="F" else cS_io),
                    height=0.62, edgecolor="white", linewidth=0.8)
        for (t0, dt, kind) in s0_cmp:
            ax.barh(y_pos[1], dt, left=t0, color=(cF_cmp if kind=="F" else cS_cmp),
                    height=0.62, edgecolor="white", linewidth=0.8)
        for (t0, dt, kind) in s1_io:
            ax.barh(y_pos[2], dt, left=t0, color=(cF_io if kind=="F" else cS_io),
                    height=0.62, edgecolor="white", linewidth=0.8)
        for (t0, dt, kind) in s1_cmp:
            ax.barh(y_pos[3], dt, left=t0, color=(cF_cmp if kind=="F" else cS_cmp),
                    height=0.62, edgecolor="white", linewidth=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(lane_labels, fontsize=10)
        ax.set_xlim(0, max_t)
        ax.set_ylim(-0.6, 4.1)
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(title, fontsize=12, loc="left", pad=4)

        # End-of-pipeline marker
        ax.axvline(total, color="#222", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(total + 0.25, 1.5, f"step ends\n{total:.2f} ms", fontsize=10,
                color="#222", fontweight="bold", va="center", ha="left")

        # Stage 0 end marker (when stage 1 compute can begin)
        ax.axvline(s0_end, color="#888", linestyle=":", linewidth=1, alpha=0.6)
        ax.text(s0_end, 3.7, f"GPU 0 done @ {s0_end:.2f} ↓",
                fontsize=8, color="#555", ha="center", va="bottom")

    axes[1].set_xlabel("time (ms)", fontsize=11)

    # Speedup badge between panels
    fig.text(0.5, 0.498, f"IndexCache:  {speedup:.2f}× faster",
             ha="center", va="center", fontsize=13, fontweight="bold",
             color="#1e6e1e",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f3e8",
                       edgecolor="#1e6e1e", linewidth=1.2))

    handles = [
        mpatches.Patch(color=cF_io,  label="IO — F layer (indexer-K + KV)"),
        mpatches.Patch(color=cS_io,  label="IO — S layer (KV only)"),
        mpatches.Patch(color=cF_cmp, label="Compute — F layer"),
        mpatches.Patch(color=cS_cmp, label="Compute — S layer"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Decode-step schedule — HBM mode  "
                 "(illustrative: 2 GPU ranks × 8 layers each, BS=1 sl=4M FP16)",
                 fontsize=13, y=0.99)

    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    path = os.path.join(OUT, "fig_pipeline_timeline.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path}")


# ─────────────────────────────────────────────────────────────────────
# Figure 4b — PP=4 FP8 pipeline timeline
# ─────────────────────────────────────────────────────────────────────

def fig_pipeline_timeline_pp4_fp8():
    """HBM-mode pipeline Gantt for PP=4 FP8: DSA vs IndexCache.

    4 GPU ranks, each holding 4 layers (illustrative: actual PP=4 has ~15/rank).
    Real per-layer costs from analytical_layer at BS=1 sl=4M FP8 HBM.
    """
    f = analytical_layer_F(4 * 1024 * 1024, 1, "hbm", ep=8, fp8=True)
    s = analytical_layer_S(4 * 1024 * 1024, 1, "hbm", ep=8, fp8=True)
    F_io  = f.total_io_ms()
    S_io  = s.total_io_ms()
    F_cmp = f.total_compute_ms(tp=1)
    S_cmp = s.total_compute_ms(tp=1)

    # 4 GPUs × 4 layers each = 16 layers total, F:S:S:S repeating
    dsa_pattern = ["F"] * 4
    ic_pattern  = ["F", "S", "S", "S"]

    def simulate_pp4(pat):
        """PP=4: GPU 0 runs producer/consumer; GPUs 1-3 each start compute
        at max(prev_gpu_done, this_gpu_io_done). All IO buses run from t=0."""
        # GPU 0: interleaved IO + compute
        cum_io, end_cmp = 0.0, 0.0
        gpu_io = [[]]
        gpu_cmp = [[]]
        gpu_io_ends = []
        for kind in pat:
            dt_io  = F_io  if kind == "F" else S_io
            dt_cmp = F_cmp if kind == "F" else S_cmp
            gpu_io[0].append((cum_io, dt_io, kind))
            cum_io += dt_io
            cmp_start = max(end_cmp, cum_io)
            gpu_cmp[0].append((cmp_start, dt_cmp, kind))
            end_cmp = cmp_start + dt_cmp
        gpu_io_ends.append(cum_io)
        prev_end = end_cmp

        # GPUs 1-3: IO prefetches from t=0 on independent buses
        for _ in range(1, 4):
            io_bars = []
            cum_io = 0.0
            for kind in pat:
                dt_io = F_io if kind == "F" else S_io
                io_bars.append((cum_io, dt_io, kind))
                cum_io += dt_io
            gpu_io.append(io_bars)
            gpu_io_ends.append(cum_io)

            cmp_t = max(prev_end, cum_io)
            cmp_bars = []
            for kind in pat:
                dt_cmp = F_cmp if kind == "F" else S_cmp
                cmp_bars.append((cmp_t, dt_cmp, kind))
                cmp_t += dt_cmp
            gpu_cmp.append(cmp_bars)
            prev_end = cmp_t

        return gpu_io, gpu_cmp, gpu_io_ends, prev_end

    dsa_io, dsa_cmp, dsa_io_ends, dsa_total = simulate_pp4(dsa_pattern)
    ic_io,  ic_cmp,  ic_io_ends,  ic_total  = simulate_pp4(ic_pattern)
    speedup = dsa_total / ic_total

    cF_io  = "#c44e52"
    cS_io  = "#e8b554"
    cF_cmp = "#2a4a8b"
    cS_cmp = "#7aa3d0"

    n_lanes = 8  # 4 GPUs × (IO + Compute)
    lane_labels = []
    for i in range(4):
        lane_labels += [f"GPU {i} · IO bus", f"GPU {i} · Compute"]
    # y positions: GPU 0 at top (y=7,6), GPU 1 (y=5,4), etc.
    y_io  = [7, 5, 3, 1]
    y_cmp = [6, 4, 2, 0]
    y_all = sorted(y_io + y_cmp, reverse=True)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw={"hspace": 0.45})
    titles = [
        f"DSA  ·  all 16 layers are F  ·  TPOT = {dsa_total:.2f} ms",
        f"IndexCache F:S:S:S  ·  4 F + 12 S  ·  TPOT = {ic_total:.2f} ms",
    ]
    max_t = max(dsa_total, ic_total) * 1.12

    for ax, g_io, g_cmp, g_io_ends, total in [
        (axes[0], dsa_io, dsa_cmp, dsa_io_ends, dsa_total),
        (axes[1], ic_io,  ic_cmp,  ic_io_ends,  ic_total),
    ]:
        for i in range(4):
            for (t0, dt, kind) in g_io[i]:
                ax.barh(y_io[i], dt, left=t0,
                        color=(cF_io if kind == "F" else cS_io),
                        height=0.62, edgecolor="white", linewidth=0.8)
            for (t0, dt, kind) in g_cmp[i]:
                ax.barh(y_cmp[i], dt, left=t0,
                        color=(cF_cmp if kind == "F" else cS_cmp),
                        height=0.62, edgecolor="white", linewidth=0.8)
            # Dotted line: when GPU i-1 finishes (GPU i compute start gate)
            if i > 0:
                prev_end = g_cmp[i-1][-1][0] + g_cmp[i-1][-1][1]
                ax.axvline(prev_end, color="#aaa", linestyle=":", linewidth=0.9, alpha=0.7)
                ax.text(prev_end, y_io[i] + 0.5,
                        f"GPU {i-1} done\n@ {prev_end:.2f}",
                        fontsize=7, color="#666", ha="center", va="bottom")

        ax.set_yticks(y_all)
        ax.set_yticklabels(lane_labels, fontsize=9)
        ax.set_xlim(0, max_t)
        ax.set_ylim(-0.6, 8.6)
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(titles[0] if ax is axes[0] else titles[1],
                     fontsize=11, loc="left", pad=4)

        ax.axvline(total, color="#222", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(total + 0.15, 3.5, f"step ends\n{total:.2f} ms",
                fontsize=10, color="#222", fontweight="bold",
                va="center", ha="left")

    axes[1].set_xlabel("time (ms)", fontsize=11)

    fig.text(0.5, 0.498, f"IndexCache:  {speedup:.2f}× faster",
             ha="center", va="center", fontsize=13, fontweight="bold",
             color="#1e6e1e",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f3e8",
                       edgecolor="#1e6e1e", linewidth=1.2))

    handles = [
        mpatches.Patch(color=cF_io,  label="IO — F layer (indexer-K + KV)"),
        mpatches.Patch(color=cS_io,  label="IO — S layer (KV only)"),
        mpatches.Patch(color=cF_cmp, label="Compute — F layer"),
        mpatches.Patch(color=cS_cmp, label="Compute — S layer"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Decode-step schedule — PP=4 FP8 HBM mode  "
                 "(illustrative: 4 GPU ranks × 4 layers each, BS=1 sl=4M)",
                 fontsize=13, y=0.99)

    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    path = os.path.join(OUT, "fig_pipeline_timeline_pp4_fp8.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
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
    fig_pipeline_timeline_pp4_fp8()
    fig_tpot_vs_sl()
    fig_speedup_hbm_vs_offload()
