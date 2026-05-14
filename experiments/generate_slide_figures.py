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
# Figure 4 — Pipeline Gantt: one IO + one Compute lane per PP stage
# (TP is abstracted away — compute uses TP-sharded times)
# ─────────────────────────────────────────────────────────────────────
def _simulate_pp_layers(pat, pp, F_io, S_io, F_cmp, S_cmp):
    """Per-stage layer schedule. Stage 0 interleaves IO+compute.
    Stages 1..PP-1 start compute at max(prev_stage_end, this_io_end).
    Returns (stages, total) where stages[s] is list of
    (kind, cmp_s, cmp_e, io_s, io_e)."""
    stages = []
    cum_io, end_cmp = 0.0, 0.0
    s0 = []
    for kind in pat:
        dt_io  = F_io  if kind == "F" else S_io
        dt_cmp = F_cmp if kind == "F" else S_cmp
        io_s, io_e = cum_io, cum_io + dt_io
        cum_io = io_e
        cmp_s = max(end_cmp, cum_io)
        cmp_e = cmp_s + dt_cmp
        end_cmp = cmp_e
        s0.append((kind, cmp_s, cmp_e, io_s, io_e))
    stages.append(s0)
    prev_end = end_cmp
    for _ in range(1, pp):
        cum_io = 0.0
        io_list = []
        for kind in pat:
            dt_io = F_io if kind == "F" else S_io
            io_list.append((cum_io, cum_io + dt_io))
            cum_io += dt_io
        cmp_t = max(prev_end, cum_io)
        sk = []
        for i, kind in enumerate(pat):
            dt_cmp = F_cmp if kind == "F" else S_cmp
            sk.append((kind, cmp_t, cmp_t + dt_cmp,
                       io_list[i][0], io_list[i][1]))
            cmp_t += dt_cmp
        stages.append(sk)
        prev_end = cmp_t
    return stages, prev_end


def _draw_pp_gantt(ax, stages, total, pp, max_t, title):
    cF_io, cS_io  = "#c44e52", "#e8b554"
    cF_cmp, cS_cmp = "#2a4a8b", "#7aa3d0"

    # Two lanes per PP stage: IO (top) and Compute (bottom).
    # PP0 lanes at top of figure.
    def lane_y(stage, kind):  # kind="io" or "cmp"
        base = (pp - 1 - stage) * 2.5
        return base + (1.0 if kind == "io" else 0.0)

    bar_h = 0.7
    for stage_idx, layers in enumerate(stages):
        for (kind, cmp_s, cmp_e, io_s, io_e) in layers:
            ax.barh(lane_y(stage_idx, "io"), io_e - io_s, left=io_s,
                    color=cF_io if kind == "F" else cS_io,
                    height=bar_h, edgecolor="white", linewidth=0.6)
            ax.barh(lane_y(stage_idx, "cmp"), cmp_e - cmp_s, left=cmp_s,
                    color=cF_cmp if kind == "F" else cS_cmp,
                    height=bar_h, edgecolor="white", linewidth=0.6)

    yticks, ylabels = [], []
    for stage in range(pp):
        yticks += [lane_y(stage, "io"), lane_y(stage, "cmp")]
        ylabels += [f"PP{stage} · IO", f"PP{stage} · Compute"]
    pairs = sorted(zip(yticks, ylabels), key=lambda p: -p[0])
    ax.set_yticks([p[0] for p in pairs])
    ax.set_yticklabels([p[1] for p in pairs], fontsize=10)

    ax.set_ylim(-0.7, lane_y(0, "io") + 0.9)
    ax.set_xlim(0, max_t)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(title, loc="left", fontsize=11.5, pad=4)

    for stage in range(pp - 1):
        stage_end = stages[stage][-1][2]
        ax.axvline(stage_end, color="#888", linestyle=":",
                   linewidth=0.9, alpha=0.75)

    ax.axvline(total, color="#222", linestyle="--",
               linewidth=1.3, alpha=0.9)
    ax.text(total + max_t * 0.006, lane_y(0, "io") / 2,
            f"TPOT\n{total:.2f} ms",
            fontsize=10, color="#222", fontweight="bold",
            va="center", ha="left")


def _save_pp_pipeline(pp, tp, fp8, layers_per_stage, sl, bs, fname, label):
    f = analytical_layer_F(sl, bs, "hbm", ep=8, fp8=fp8)
    s = analytical_layer_S(sl, bs, "hbm", ep=8, fp8=fp8)
    F_io, S_io = f.total_io_ms(), s.total_io_ms()
    F_cmp = f.total_compute_ms(tp=tp)
    S_cmp = s.total_compute_ms(tp=tp)

    dsa_pat = ["F"] * layers_per_stage
    ic_pat  = (["F", "S", "S", "S"] * ((layers_per_stage + 3) // 4))[:layers_per_stage]

    dsa_stages, dsa_total = _simulate_pp_layers(dsa_pat, pp, F_io, S_io, F_cmp, S_cmp)
    ic_stages,  ic_total  = _simulate_pp_layers(ic_pat,  pp, F_io, S_io, F_cmp, S_cmp)
    speedup = dsa_total / ic_total

    n_total = pp * layers_per_stage
    n_F_ic  = sum(1 for x in ic_pat if x == "F") * pp
    n_S_ic  = n_total - n_F_ic

    max_t = max(dsa_total, ic_total) * 1.14
    fig_h = max(5.5, 0.55 * pp * 2 + 3.0)
    fig, axes = plt.subplots(2, 1, figsize=(12, fig_h), sharex=True,
                              gridspec_kw={"hspace": 0.5})

    _draw_pp_gantt(axes[0], dsa_stages, dsa_total, pp, max_t,
                   f"DSA  ·  all {n_total} layers F  ·  TPOT = {dsa_total:.2f} ms")
    _draw_pp_gantt(axes[1], ic_stages, ic_total, pp, max_t,
                   f"IndexCache F:S:S:S  ·  {n_F_ic} F + {n_S_ic} S  ·  TPOT = {ic_total:.2f} ms")
    axes[1].set_xlabel("time (ms)", fontsize=11)

    fig.text(0.5, 0.498, f"IndexCache:  {speedup:.2f}× faster",
             ha="center", va="center", fontsize=13, fontweight="bold",
             color="#1e6e1e",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f3e8",
                       edgecolor="#1e6e1e", linewidth=1.2))

    cF_io, cS_io  = "#c44e52", "#e8b554"
    cF_cmp, cS_cmp = "#2a4a8b", "#7aa3d0"
    handles = [
        mpatches.Patch(color=cF_io,  label="IO — F (indexer-K + KV)"),
        mpatches.Patch(color=cS_io,  label="IO — S (KV only)"),
        mpatches.Patch(color=cF_cmp, label="Compute — F layer"),
        mpatches.Patch(color=cS_cmp, label="Compute — S layer"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"Decode-step schedule — {label}\n"
                 f"(Each PP lane = group of {tp} TP GPUs working in parallel; "
                 "see separate TP figure)",
                 fontsize=11.5, y=0.995)

    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = os.path.join(OUT, fname)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path}")


def fig_pipeline_timeline():
    """PP=2 FP16 HBM, BS=1 sl=4M."""
    _save_pp_pipeline(pp=2, tp=2, fp8=False, layers_per_stage=4,
                      sl=4 * 1024 * 1024, bs=1,
                      fname="fig_pipeline_timeline.png",
                      label="PP=2 TP=2 FP16 HBM  (illustrative: 4 layers/stage, BS=1 sl=4M)")


def fig_pipeline_timeline_pp4_fp8():
    """PP=4 FP8 HBM, BS=1 sl=4M."""
    _save_pp_pipeline(pp=4, tp=4, fp8=True, layers_per_stage=4,
                      sl=4 * 1024 * 1024, bs=1,
                      fname="fig_pipeline_timeline_pp4_fp8.png",
                      label="PP=4 TP=4 FP8 HBM  (illustrative: 4 layers/stage, BS=1 sl=4M)")


# ─────────────────────────────────────────────────────────────────────
# Figure 4c — Tensor Parallelism (TP) visualization
# ─────────────────────────────────────────────────────────────────────
def fig_tp_visualization():
    """Show how TP=4 GPUs cooperate on one layer in parallel.

    Inside one PP stage, TP GPUs all execute the same layer at the same
    time, each holding 1/TP of the weight shards. They synchronize at
    all-reduce boundaries (after Q/K/V proj, after attention out-proj,
    after MoE).
    """
    TP = 4
    # Per-piece timing (FP8, BS=1 sl=4M, scaled by 1/TP for shardable parts)
    # Numbers picked illustratively from analytical_layer F-layer breakdown.
    pieces = [
        ("block_a (Q proj)",   0.20, True,  "#4c8bb0"),
        ("AllReduce",          0.04, False, "#aaaaaa"),
        ("Attention + o_proj", 0.60, True,  "#5a9c5e"),
        ("AllReduce",          0.04, False, "#aaaaaa"),
        ("MoE shardable",      0.30, True,  "#8ac270"),
        ("MoE expert + EP",    0.28, False, "#7568b0"),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    bar_h = 0.55
    y_positions = list(range(TP - 1, -1, -1))  # GPU 0 at top

    t = 0.0
    for name, dt, sharded, color in pieces:
        for y in y_positions:
            ax.barh(y, dt, left=t, color=color,
                    height=bar_h, edgecolor="white", linewidth=0.7)
        if dt > 0.06:
            ax.text(t + dt / 2, TP - 0.3, name,
                    ha="center", va="bottom", fontsize=9, color="#333")
            if sharded:
                ax.text(t + dt / 2, -0.6, "(÷TP)",
                        ha="center", va="top", fontsize=8, color="#666",
                        fontstyle="italic")
        t += dt

    total = t
    ax.axvline(total, color="#222", linestyle="--", linewidth=1.3)
    ax.text(total + 0.02, TP / 2 - 0.5, f"layer end\n{total:.2f} ms",
            fontsize=10, fontweight="bold", color="#222", va="center")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"GPU {i}  (TP rank {i})" for i in range(TP)],
                       fontsize=10, family="monospace")
    ax.set_xlabel("time within one layer (ms)", fontsize=11)
    ax.set_xlim(0, total * 1.18)
    ax.set_ylim(-1.1, TP + 0.6)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Tensor Parallelism — TP=4 GPUs cooperate on one layer\n"
                 "All 4 GPUs run the same pieces at the same time. "
                 "Shardable pieces are divided by TP; AllReduce after each block.",
                 fontsize=11, loc="left", pad=8)

    plt.tight_layout()
    path = os.path.join(OUT, "fig_tp_visualization.png")
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


# ─────────────────────────────────────────────────────────────────────
# Figure 6 — KV-on-SSD speedup bars  (PP=2 TP=2 EP=8 FP8)
# ─────────────────────────────────────────────────────────────────────
def _kv_ssd_speedup(seq_len, bs, ssd_bw):
    from experiments.dsa_vs_ic_kv_ssd import (
        build_pattern as build_ssd, schedule_pp as sched_ssd)
    dsa = sched_ssd(build_ssd("DSA", seq_len, bs, 2, True, 8, ssd_bw), 2)
    ic  = sched_ssd(build_ssd("IC",  seq_len, bs, 2, True, 8, ssd_bw), 2)
    return dsa / ic, dsa, ic


def fig_speedup_kv_ssd():
    """Grouped bars: IC speedup with KV-on-SSD at 7 GB/s vs 28 GB/s."""
    corners = [
        ("200K\nBS=1",  200*1024,        1),
        ("1M\nBS=1",    1024*1024,       1),
        ("2M\nBS=1",    2*1024*1024,     1),
        ("4M\nBS=1",    4*1024*1024,     1),
        ("128K\nBS=8",  128*1024,        8),
        ("200K\nBS=8",  200*1024,        8),
        ("512K\nBS=8",  512*1024,        8),
        ("1M\nBS=8",    1024*1024,       8),
        ("2M\nBS=8",    2*1024*1024,     8),
        ("200K\nBS=32", 200*1024,       32),
        ("512K\nBS=32", 512*1024,       32),
    ]
    sp_7,  sp_28 = [], []
    for _, sl, bs in corners:
        sp_7.append (_kv_ssd_speedup(sl, bs,  7.0)[0])
        sp_28.append(_kv_ssd_speedup(sl, bs, 28.0)[0])
    labels = [c[0] for c in corners]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars_a = ax.bar(x - w/2, sp_7,  w, label="Gen4 single NVMe (7 GB/s)",  color="#bf6b3a")
    bars_b = ax.bar(x + w/2, sp_28, w, label="4× Gen4 NVMe RAID (28 GB/s)", color="#3b6bb0")
    ax.axhline(1.0, color="#666", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(labels) - 0.5, 1.05, "1.0× = no change",
            color="#666", fontsize=9, ha="right")

    for bars in (bars_a, bars_b):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03,
                    f"{b.get_height():.2f}×", ha="center", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("IndexCache speedup  (DSA TPOT / IC TPOT)")
    ax.set_title("KV-on-SSD: IndexCache speedup vs DSA  ·  PP=2 TP=2 EP=8 FP8")
    ax.set_ylim(0.9, max(max(sp_7), max(sp_28)) * 1.12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(OUT, "fig_speedup_kv_ssd.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"  wrote {path}")


def fig_tpot_vs_sl_kv_ssd():
    """TPOT vs seq_len at BS=8 for KV-on-SSD; one panel each per SSD speed."""
    sls = [32*1024, 128*1024, 200*1024, 512*1024, 1024*1024, 2*1024*1024]
    sl_lbls = ["32K", "128K", "200K", "512K", "1M", "2M"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, (bw, ttl) in zip(axes, [(7.0, "Gen4 NVMe single drive (7 GB/s)"),
                                    (28.0, "4× Gen4 NVMe RAID (28 GB/s)")]):
        dsa_y, ic_y = [], []
        for sl in sls:
            _, dsa, ic = _kv_ssd_speedup(sl, 8, bw)
            dsa_y.append(dsa)
            ic_y.append(ic)
        ax.plot(sls, dsa_y, "o--", color="#c44e52",
                label="DSA",        linewidth=2.2, markersize=7)
        ax.plot(sls, ic_y,  "o-",  color="#2a4a8b",
                label="IndexCache", linewidth=2.5, markersize=7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sls)
        ax.set_xticklabels(sl_lbls)
        ax.set_xlabel("seq_len (log scale)")
        ax.set_title(ttl, fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper left", fontsize=10)

    axes[0].set_ylabel("TPOT (ms / output token)")
    fig.suptitle("KV-on-SSD TPOT vs seq_len at BS=8  ·  PP=2 TP=2 EP=8 FP8",
                 fontsize=12.5, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT, "fig_tpot_vs_sl_kv_ssd_bs8.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path}")


if __name__ == "__main__":
    print(f"Writing slide figures to {OUT}/")
    fig_fs_schematic()
    fig_f_vs_s_breakdown()
    fig_pipeline_timeline()
    fig_pipeline_timeline_pp4_fp8()
    fig_tp_visualization()
    fig_tpot_vs_sl()
    fig_speedup_hbm_vs_offload()
    fig_speedup_kv_ssd()
    fig_tpot_vs_sl_kv_ssd()
