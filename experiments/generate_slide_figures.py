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
def _simulate_pp(pat, pp, F_io, S_io, F_cmp, S_cmp):
    """Simulate PP-stage timing for a layer pattern.

    Returns list of (stage_idx, layer_kind, cmp_start, cmp_end, io_start, io_end)
    for every layer on every stage. IO always starts at t=0 per stage (independent
    HBM buses). Stage 0 interleaves IO+compute; stages 1..PP-1 start compute at
    max(prev_stage_end, this_stage_io_end).
    """
    layers_per_stage = len(pat)
    records = []

    # Stage 0: interleaved IO + compute (producer/consumer)
    cum_io, end_cmp = 0.0, 0.0
    for kind in pat:
        dt_io  = F_io  if kind == "F" else S_io
        dt_cmp = F_cmp if kind == "F" else S_cmp
        io_s, io_e = cum_io, cum_io + dt_io
        cum_io = io_e
        cmp_s = max(end_cmp, cum_io)
        cmp_e = cmp_s + dt_cmp
        end_cmp = cmp_e
        records.append((0, kind, cmp_s, cmp_e, io_s, io_e))
    prev_end = end_cmp

    for stage in range(1, pp):
        cum_io = 0.0
        io_recs = []
        for kind in pat:
            dt_io = F_io if kind == "F" else S_io
            io_recs.append((cum_io, cum_io + dt_io))
            cum_io += dt_io
        stage_io_end = cum_io

        cmp_t = max(prev_end, stage_io_end)
        for idx, kind in enumerate(pat):
            dt_cmp = F_cmp if kind == "F" else S_cmp
            io_s, io_e = io_recs[idx]
            records.append((stage, kind, cmp_t, cmp_t + dt_cmp, io_s, io_e))
            cmp_t += dt_cmp
        prev_end = cmp_t

    return records, prev_end


def _draw_per_gpu_gantt(ax, records, total, pp, tp, max_t,
                        cF_cmp, cS_cmp, cF_io, cS_io, title):
    """Draw one Gantt panel with one lane per GPU.

    Each PP stage has `tp` GPU lanes. TP GPUs within a stage are in sync
    (identical compute windows) — shown as parallel identical bars.
    IO is drawn as a thin bar just below each GPU's compute bar.
    """
    # y layout: GPU 0 at top. Each GPU gets a height-1 slot.
    # Within each PP stage the TP GPUs are adjacent, separated by a small gap
    # between stages.
    n_gpus = pp * tp
    gap = 0.4  # extra gap between PP stage groups

    def gpu_y(stage, tp_rank):
        # y increases downward in barh convention (we'll invert axis)
        return (pp - 1 - stage) * (tp + gap) + (tp - 1 - tp_rank)

    bar_h   = 0.75
    io_h    = 0.18
    io_offset = -0.47  # below the compute bar

    for (stage, kind, cmp_s, cmp_e, io_s, io_e) in records:
        for t in range(tp):
            y = gpu_y(stage, t)
            # Compute bar
            ax.barh(y, cmp_e - cmp_s, left=cmp_s,
                    color=(cF_cmp if kind == "F" else cS_cmp),
                    height=bar_h, edgecolor="white", linewidth=0.5)
            # IO bar (thin, below compute)
            ax.barh(y + io_offset, io_e - io_s, left=io_s,
                    color=(cF_io if kind == "F" else cS_io),
                    height=io_h, edgecolor="none")

    # Y-tick labels: one per GPU
    yticks, ylabels = [], []
    for stage in range(pp - 1, -1, -1):
        for t in range(tp - 1, -1, -1):
            y = gpu_y(stage, t)
            yticks.append(y)
            ylabels.append(f"GPU{stage*tp+t}  (PP{stage},TP{t})")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8.5, family="monospace")
    ax.set_xlim(0, max_t)
    y_min = gpu_y(0, tp - 1) - 0.7
    y_max = gpu_y(pp - 1, 0) + 0.7
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(title, fontsize=11, loc="left", pad=4)

    # Dashed vertical line at step end
    ax.axvline(total, color="#222", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.text(total + max_t * 0.005, (y_min + y_max) / 2,
            f"TPOT\n{total:.2f} ms",
            fontsize=9, color="#222", fontweight="bold",
            va="center", ha="left")

    # PP stage boundary lines and labels on left
    for stage in range(pp):
        y_top = gpu_y(stage, 0) + bar_h / 2 + 0.05
        y_bot = gpu_y(stage, tp - 1) - bar_h / 2 - 0.05
        ax.annotate("", xy=(0, y_bot), xytext=(0, y_top),
                    xycoords="data", textcoords="data",
                    arrowprops=dict(arrowstyle="-", color="#bbb", lw=1.2))

    # Dotted stage-handoff lines
    for stage in range(pp - 1):
        stage_end = max(r[3] for r in records if r[0] == stage)
        gate_y = gpu_y(stage, tp - 1) - 0.5
        ax.axvline(stage_end, color="#aaa", linestyle=":", linewidth=0.9, alpha=0.7)
        ax.text(stage_end, gate_y,
                f"↓ PP{stage} done\n  {stage_end:.2f} ms",
                fontsize=7, color="#666", ha="left", va="top")


def fig_pipeline_timeline():
    """Per-GPU Gantt: PP=2 TP=2 FP16 HBM, DSA vs IndexCache.

    4 GPUs total (GPU0-1 = PP stage 0, GPU2-3 = PP stage 1).
    TP GPUs in the same stage work on the same layers at the same time.
    Illustrative: 2 PP stages × 4 layers each. BS=1 sl=4M FP16.
    """
    PP, TP = 2, 2
    f = analytical_layer_F(4 * 1024 * 1024, 1, "hbm", ep=8, fp8=False)
    s = analytical_layer_S(4 * 1024 * 1024, 1, "hbm", ep=8, fp8=False)
    F_io, S_io = f.total_io_ms(), s.total_io_ms()
    F_cmp = f.total_compute_ms(tp=TP)
    S_cmp = s.total_compute_ms(tp=TP)

    dsa_pat = ["F", "F", "F", "F"]
    ic_pat  = ["F", "S", "S", "S"]

    dsa_rec, dsa_total = _simulate_pp(dsa_pat, PP, F_io, S_io, F_cmp, S_cmp)
    ic_rec,  ic_total  = _simulate_pp(ic_pat,  PP, F_io, S_io, F_cmp, S_cmp)
    speedup = dsa_total / ic_total

    cF_io  = "#c44e52"; cS_io  = "#e8b554"
    cF_cmp = "#2a4a8b"; cS_cmp = "#7aa3d0"

    max_t = max(dsa_total, ic_total) * 1.13
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                              gridspec_kw={"hspace": 0.55})

    _draw_per_gpu_gantt(axes[0], dsa_rec, dsa_total, PP, TP, max_t,
                        cF_cmp, cS_cmp, cF_io, cS_io,
                        f"DSA  ·  all 8 layers F  ·  TPOT = {dsa_total:.2f} ms")
    _draw_per_gpu_gantt(axes[1], ic_rec, ic_total, PP, TP, max_t,
                        cF_cmp, cS_cmp, cF_io, cS_io,
                        f"IndexCache F:S:S:S  ·  2 F + 6 S  ·  TPOT = {ic_total:.2f} ms")

    axes[1].set_xlabel("time (ms)", fontsize=11)

    fig.text(0.5, 0.498, f"IndexCache:  {speedup:.2f}× faster",
             ha="center", va="center", fontsize=13, fontweight="bold", color="#1e6e1e",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f3e8",
                       edgecolor="#1e6e1e", linewidth=1.2))

    handles = [
        mpatches.Patch(color=cF_cmp, label="Compute — F layer"),
        mpatches.Patch(color=cS_cmp, label="Compute — S layer"),
        mpatches.Patch(color=cF_io,  label="IO — F layer (thin bar)"),
        mpatches.Patch(color=cS_io,  label="IO — S layer (thin bar)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Decode-step schedule — PP=2 TP=2 FP16 HBM  "
                 "(illustrative: 2 PP stages × 4 layers, BS=1 sl=4M)",
                 fontsize=12, y=0.995)

    plt.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = os.path.join(OUT, "fig_pipeline_timeline.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path}")


# ─────────────────────────────────────────────────────────────────────
# Figure 4b — PP=4 TP=4 FP8 pipeline timeline
# ─────────────────────────────────────────────────────────────────────

def fig_pipeline_timeline_pp4_fp8():
    """Per-GPU Gantt: PP=4 TP=4 FP8 HBM, DSA vs IndexCache.

    16 GPUs total (GPU0-3 = PP stage 0, GPU4-7 = PP stage 1, …).
    TP GPUs in the same stage work on the same layers at the same time.
    Illustrative: 4 PP stages × 4 layers each. BS=1 sl=4M FP8.
    """
    PP, TP = 4, 4
    f = analytical_layer_F(4 * 1024 * 1024, 1, "hbm", ep=8, fp8=True)
    s = analytical_layer_S(4 * 1024 * 1024, 1, "hbm", ep=8, fp8=True)
    F_io, S_io = f.total_io_ms(), s.total_io_ms()
    F_cmp = f.total_compute_ms(tp=TP)
    S_cmp = s.total_compute_ms(tp=TP)

    dsa_pat = ["F", "F", "F", "F"]
    ic_pat  = ["F", "S", "S", "S"]

    dsa_rec, dsa_total = _simulate_pp(dsa_pat, PP, F_io, S_io, F_cmp, S_cmp)
    ic_rec,  ic_total  = _simulate_pp(ic_pat,  PP, F_io, S_io, F_cmp, S_cmp)
    speedup = dsa_total / ic_total

    cF_io  = "#c44e52"; cS_io  = "#e8b554"
    cF_cmp = "#2a4a8b"; cS_cmp = "#7aa3d0"

    max_t = max(dsa_total, ic_total) * 1.13
    fig, axes = plt.subplots(2, 1, figsize=(13, 11), sharex=True,
                              gridspec_kw={"hspace": 0.45})

    _draw_per_gpu_gantt(axes[0], dsa_rec, dsa_total, PP, TP, max_t,
                        cF_cmp, cS_cmp, cF_io, cS_io,
                        f"DSA  ·  all 16 layers F  ·  TPOT = {dsa_total:.2f} ms")
    _draw_per_gpu_gantt(axes[1], ic_rec, ic_total, PP, TP, max_t,
                        cF_cmp, cS_cmp, cF_io, cS_io,
                        f"IndexCache F:S:S:S  ·  4 F + 12 S  ·  TPOT = {ic_total:.2f} ms")

    axes[1].set_xlabel("time (ms)", fontsize=11)

    fig.text(0.5, 0.498, f"IndexCache:  {speedup:.2f}× faster",
             ha="center", va="center", fontsize=13, fontweight="bold", color="#1e6e1e",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f3e8",
                       edgecolor="#1e6e1e", linewidth=1.2))

    handles = [
        mpatches.Patch(color=cF_cmp, label="Compute — F layer"),
        mpatches.Patch(color=cS_cmp, label="Compute — S layer"),
        mpatches.Patch(color=cF_io,  label="IO — F layer (thin bar)"),
        mpatches.Patch(color=cS_io,  label="IO — S layer (thin bar)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Decode-step schedule — PP=4 TP=4 FP8 HBM  "
                 "(illustrative: 4 PP stages × 4 layers, BS=1 sl=4M)",
                 fontsize=12, y=0.995)

    plt.tight_layout(rect=(0, 0.05, 1, 0.97))
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
