#!/usr/bin/env python3
"""DSA batch-size sweep.

Answers the question: does increasing batch size make offloading (PCIe)
competitive with all-in-memory (HBM)?

For each (seq_len, mode) we sweep BS from 1 to 256 and report two
distinct metrics, which point in opposite directions:

  1. TPOT (per-sequence latency)  = layer_total × 61
     This is what each user waits between successive tokens. It
     INCREASES with BS because the batched decode step does more work.

  2. Throughput (tokens/sec)  = BS / (TPOT/1000)
     Aggregate system token rate across the batch. INCREASES with BS
     until the shared bottleneck (weight-read HBM or indexer-K PCIe)
     saturates.

The "overlap" question is really about throughput: does offload's
throughput curve catch up to HBM's when we batch? Answer below.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.dsa_layer_analyzer import analyze_layer_bs, run_bs_sweep
from experiments.dsa_timing_model import load_profiles

SEQ_LENS = [4096, 32768, 128 * 1024, 512 * 1024, 1024 * 1024]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
MODES = ("hbm", "offload")

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reports", "figures")


def sl_label(sl):
    return f"{sl // 1024}K" if sl < 1024 * 1024 else f"{sl // (1024 * 1024)}M"


def ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════
# Plots
# ════════════════════════════════════════════════════════════════════

def plot_tpot_vs_bs(results):
    """Per-sequence TPOT (real latency) — increases with BS."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    axes_flat = axes.flatten()

    for idx, sl in enumerate(SEQ_LENS):
        ax = axes_flat[idx]
        for mode, color, marker in [("hbm", "#2b7bba", "o"),
                                     ("offload", "#e07b39", "s")]:
            tpots = [results[(mode, sl, bs)]["step_latency_ms"] for bs in BATCH_SIZES]
            label = "all-in-memory (HBM)" if mode == "hbm" else "offload (PCIe)"
            ax.plot(BATCH_SIZES, tpots, f"{marker}-", color=color,
                    linewidth=1.8, markersize=5, label=label)

        ax.set_xlabel("batch size")
        ax.set_ylabel("per-sequence TPOT (ms)")
        ax.set_title(f"seq_len = {sl_label(sl)}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(fontsize=8)

    axes_flat[5].set_visible(False)
    fig.suptitle("Per-sequence TPOT vs. batch size (real user latency)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "dsa_tpot_vs_bs.png"), dpi=150)
    plt.close(fig)


def plot_throughput_vs_bs(results):
    """System throughput in tokens/sec — the right metric for "does BS help"."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    axes_flat = axes.flatten()

    for idx, sl in enumerate(SEQ_LENS):
        ax = axes_flat[idx]
        for mode, color, marker in [("hbm", "#2b7bba", "o"),
                                     ("offload", "#e07b39", "s")]:
            thr = [results[(mode, sl, bs)]["throughput_tok_per_s"] for bs in BATCH_SIZES]
            label = "all-in-memory (HBM)" if mode == "hbm" else "offload (PCIe)"
            ax.plot(BATCH_SIZES, thr, f"{marker}-", color=color,
                    linewidth=1.8, markersize=5, label=label)

        ax.set_xlabel("batch size")
        ax.set_ylabel("throughput (tokens / sec)")
        ax.set_title(f"seq_len = {sl_label(sl)}")
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(fontsize=8, loc="lower right")

    axes_flat[5].set_visible(False)
    fig.suptitle("Aggregate throughput vs. batch size — does offload catch up?",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "dsa_throughput_vs_bs.png"), dpi=150)
    plt.close(fig)


def plot_throughput_pipelined(results):
    """Throughput under perfect inter-layer pipelining (IO overlap)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    axes_flat = axes.flatten()

    for idx, sl in enumerate(SEQ_LENS):
        ax = axes_flat[idx]
        for mode, color in [("hbm", "#2b7bba"), ("offload", "#e07b39")]:
            thr = [results[(mode, sl, bs)]["throughput_tok_per_s"]
                   for bs in BATCH_SIZES]
            thr_pipe = [results[(mode, sl, bs)]["pipelined_throughput_tok_per_s"]
                        for bs in BATCH_SIZES]
            label_base = "HBM" if mode == "hbm" else "offload"
            ax.plot(BATCH_SIZES, thr, "o--", color=color, linewidth=1,
                    markersize=3, alpha=0.4, label=f"{label_base} (no overlap)")
            ax.plot(BATCH_SIZES, thr_pipe, "s-", color=color, linewidth=2,
                    markersize=5, label=f"{label_base} (pipelined)")

        ax.set_xlabel("batch size")
        ax.set_ylabel("throughput (tokens / sec)")
        ax.set_title(f"seq_len = {sl_label(sl)}")
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(fontsize=7, loc="lower right")

    axes_flat[5].set_visible(False)
    fig.suptitle("Throughput with and without IO overlap (double-buffered PCIe)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "dsa_throughput_pipelined_vs_bs.png"),
                dpi=150)
    plt.close(fig)


def plot_io_vs_compute(results):
    """Per-layer IO time vs compute time — where is the overlap regime?"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    for idx, sl in enumerate(SEQ_LENS):
        ax = axes_flat[idx]
        for mode, colors in [
            ("hbm", {"io": "#9cc3e0", "cmp": "#2b7bba"}),
            ("offload", {"io": "#f5c082", "cmp": "#e07b39"}),
        ]:
            io_vals = [results[(mode, sl, bs)]["io_ms"] for bs in BATCH_SIZES]
            cmp_vals = [results[(mode, sl, bs)]["compute_ms"] for bs in BATCH_SIZES]
            lbl = "HBM" if mode == "hbm" else "offload"
            ax.plot(BATCH_SIZES, io_vals, "s--", color=colors["io"],
                    linewidth=1.5, markersize=4, label=f"IO ({lbl})")
            ax.plot(BATCH_SIZES, cmp_vals, "o-", color=colors["cmp"],
                    linewidth=1.5, markersize=4, label=f"compute ({lbl})")

        ax.set_xlabel("batch size")
        ax.set_ylabel("per-layer time (ms)")
        ax.set_title(f"seq_len = {sl_label(sl)}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(fontsize=7)

    axes_flat[5].set_visible(False)
    fig.suptitle("IO time vs. compute time per layer — where does IO dominate?",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "dsa_io_vs_compute_bs.png"), dpi=150)
    plt.close(fig)


def plot_offload_tpot_gap(results):
    """TPOT ratio = offload/HBM latency at each BS, one line per seq_len."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, len(SEQ_LENS)))

    for idx, sl in enumerate(SEQ_LENS):
        ratios = [results[("offload", sl, bs)]["step_latency_ms"]
                  / results[("hbm", sl, bs)]["step_latency_ms"]
                  for bs in BATCH_SIZES]
        ax.plot(BATCH_SIZES, ratios, "o-", color=colors[idx], linewidth=2,
                markersize=5, label=f"{sl_label(sl)}")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("batch size")
    ax.set_ylabel("TPOT slowdown (offload / HBM)")
    ax.set_title("Per-sequence TPOT slowdown from offloading")
    ax.set_xscale("log", base=2)
    ax.legend(title="seq_len")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dsa_offload_gap_vs_bs.png"), dpi=150)
    plt.close(fig)


def plot_pipelined_tpot_gap(results):
    """Pipelined TPOT ratio, one line per seq_len."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, len(SEQ_LENS)))

    for idx, sl in enumerate(SEQ_LENS):
        ratios = [results[("offload", sl, bs)]["pipelined_step_latency_ms"]
                  / results[("hbm", sl, bs)]["pipelined_step_latency_ms"]
                  for bs in BATCH_SIZES]
        ax.plot(BATCH_SIZES, ratios, "o-", color=colors[idx], linewidth=2,
                markersize=5, label=f"{sl_label(sl)}")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("batch size")
    ax.set_ylabel("pipelined TPOT slowdown (offload / HBM)")
    ax.set_title("Pipelined per-sequence TPOT slowdown from offloading")
    ax.set_xscale("log", base=2)
    ax.legend(title="seq_len")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dsa_pipelined_gap_vs_bs.png"), dpi=150)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# Summary tables
# ════════════════════════════════════════════════════════════════════

def print_tpot_table(results):
    print("\n" + "=" * 105)
    print(" TPOT (per-sequence decode-step latency, ms) — what each user waits")
    print(" TPOT = layer_time × 61. Grows with BS because each step does more work.")
    print("=" * 105)
    hdr = "%6s  %8s" % ("seq", "mode")
    for bs in BATCH_SIZES:
        hdr += "  BS=%-5d" % bs
    print(hdr)
    print("-" * 105)
    for sl in SEQ_LENS:
        for mode in MODES:
            row = "%6s  %8s" % (sl_label(sl), mode)
            for bs in BATCH_SIZES:
                v = results[(mode, sl, bs)]["step_latency_ms"]
                row += "  %8.1f" % v
            print(row)
        row = "%6s  %8s" % (sl_label(sl), "ratio")
        for bs in BATCH_SIZES:
            h = results[("hbm", sl, bs)]["step_latency_ms"]
            o = results[("offload", sl, bs)]["step_latency_ms"]
            row += "  %7.2fx" % (o / h)
        print(row)
        print()
    print("-" * 105)


def print_throughput_table(results):
    print("\n" + "=" * 105)
    print(" THROUGHPUT (tokens/sec aggregated across batch)")
    print(" This is the right metric for 'does batching help offload catch up?'")
    print("=" * 105)
    hdr = "%6s  %8s" % ("seq", "mode")
    for bs in BATCH_SIZES:
        hdr += "  BS=%-5d" % bs
    print(hdr)
    print("-" * 105)
    for sl in SEQ_LENS:
        for mode in MODES:
            row = "%6s  %8s" % (sl_label(sl), mode)
            for bs in BATCH_SIZES:
                v = results[(mode, sl, bs)]["throughput_tok_per_s"]
                row += "  %8.1f" % v
            print(row)
        row = "%6s  %8s" % (sl_label(sl), "ratio")
        for bs in BATCH_SIZES:
            h = results[("hbm", sl, bs)]["throughput_tok_per_s"]
            o = results[("offload", sl, bs)]["throughput_tok_per_s"]
            row += "  %7.2fx" % (o / h) if h > 0 else "     n/a"
        print(row)
        print()
    print("-" * 105)


def print_overlap_table(results):
    print("\n" + "=" * 105)
    print(" OVERLAP ANALYSIS — per-layer IO vs compute (ms), offload mode")
    print(" When compute > IO, the PCIe transfer can be fully pipelined.")
    print("=" * 105)
    hdr = "%6s  %6s" % ("seq", "")
    for bs in BATCH_SIZES:
        hdr += "  BS=%-5d" % bs
    print(hdr)
    print("-" * 105)
    for sl in SEQ_LENS:
        row_io = "%6s  %6s" % (sl_label(sl), "IO")
        row_cmp = "%6s  %6s" % ("", "comp")
        row_flag = "%6s  %6s" % ("", "")
        for bs in BATCH_SIZES:
            r = results[("offload", sl, bs)]
            row_io += "  %8.2f" % r["io_ms"]
            row_cmp += "  %8.2f" % r["compute_ms"]
            flag = "CMP>" if r["compute_ms"] > r["io_ms"] else "IO>"
            row_flag += "  %8s" % flag
        print(row_io)
        print(row_cmp)
        print(row_flag)
        print()
    print("-" * 105)


def print_pipelined_throughput_table(results):
    print("\n" + "=" * 105)
    print(" PIPELINED THROUGHPUT (tok/s) — with perfect inter-layer IO overlap")
    print("=" * 105)
    hdr = "%6s  %8s" % ("seq", "mode")
    for bs in BATCH_SIZES:
        hdr += "  BS=%-5d" % bs
    print(hdr)
    print("-" * 105)
    for sl in SEQ_LENS:
        for mode in MODES:
            row = "%6s  %8s" % (sl_label(sl), mode)
            for bs in BATCH_SIZES:
                v = results[(mode, sl, bs)]["pipelined_throughput_tok_per_s"]
                row += "  %8.1f" % v
            print(row)
        row = "%6s  %8s" % (sl_label(sl), "ratio")
        for bs in BATCH_SIZES:
            h = results[("hbm", sl, bs)]["pipelined_throughput_tok_per_s"]
            o = results[("offload", sl, bs)]["pipelined_throughput_tok_per_s"]
            row += "  %7.2fx" % (o / h) if h > 0 else "     n/a"
        print(row)
        print()
    print("-" * 105)


def main():
    ensure_dir()
    print("Running DSA batch-size sweep...")
    results = run_bs_sweep(SEQ_LENS, BATCH_SIZES, MODES)
    print(f"Computed {len(results)} configurations.\n")

    print_tpot_table(results)
    print_throughput_table(results)
    print_overlap_table(results)
    print_pipelined_throughput_table(results)

    print("\nGenerating plots...")
    plot_tpot_vs_bs(results)
    plot_throughput_vs_bs(results)
    plot_throughput_pipelined(results)
    plot_io_vs_compute(results)
    plot_offload_tpot_gap(results)
    plot_pipelined_tpot_gap(results)

    print(f"\nPlots written to {FIG_DIR}/:")
    for f in sorted(os.listdir(FIG_DIR)):
        if any(k in f for k in ("bs", "gap", "throughput", "pipelined")):
            print(f"  {f}")


if __name__ == "__main__":
    main()
