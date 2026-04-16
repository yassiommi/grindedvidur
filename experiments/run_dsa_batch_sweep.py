#!/usr/bin/env python3
"""DSA batch-size sweep: does increasing BS let PCIe IO hide behind compute?

For each (seq_len, mode) pair, sweeps BS from 1 to 256 and plots:
  1. TPOT vs BS — do the two mode lines converge?
  2. IO time vs compute time per layer — where is the crossover BS?
  3. Pipelined TPOT — what if IO could be fully overlapped with previous
     layer's compute (double-buffering)?
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


def plot_tpot_vs_bs(results):
    """TPOT vs batch size for each seq_len, both modes on same axes."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    axes_flat = axes.flatten()

    for idx, sl in enumerate(SEQ_LENS):
        ax = axes_flat[idx]
        for mode, color, marker in [("hbm", "#2b7bba", "o"), ("offload", "#e07b39", "s")]:
            tpots = [results[(mode, sl, bs)]["tpot_ms"] for bs in BATCH_SIZES]
            label = "all-in-memory (HBM)" if mode == "hbm" else "offload (PCIe)"
            ax.plot(BATCH_SIZES, tpots, f"{marker}-", color=color,
                    linewidth=1.8, markersize=5, label=label)

        ax.set_xlabel("batch size")
        ax.set_ylabel("TPOT (ms)")
        ax.set_title(f"seq_len = {sl_label(sl)}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(fontsize=8)

    # Hide the 6th subplot
    axes_flat[5].set_visible(False)

    fig.suptitle("TPOT vs. batch size — do offload and HBM converge?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "dsa_tpot_vs_bs.png"), dpi=150)
    plt.close(fig)


def plot_tpot_pipelined(results):
    """Pipelined TPOT (IO overlapped) vs batch size, both modes."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    axes_flat = axes.flatten()

    for idx, sl in enumerate(SEQ_LENS):
        ax = axes_flat[idx]
        for mode, color, ls in [("hbm", "#2b7bba", "-"), ("offload", "#e07b39", "-")]:
            tpots = [results[(mode, sl, bs)]["tpot_ms"] for bs in BATCH_SIZES]
            pipe = [results[(mode, sl, bs)]["pipelined_tpot_ms"] for bs in BATCH_SIZES]
            label_base = "HBM" if mode == "hbm" else "offload"
            ax.plot(BATCH_SIZES, tpots, "o--", color=color, linewidth=1,
                    markersize=3, alpha=0.4, label=f"{label_base} (no overlap)")
            ax.plot(BATCH_SIZES, pipe, "s-", color=color, linewidth=2,
                    markersize=5, label=f"{label_base} (pipelined)")

        ax.set_xlabel("batch size")
        ax.set_ylabel("TPOT (ms)")
        ax.set_title(f"seq_len = {sl_label(sl)}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(fontsize=7, loc="upper right")

    axes_flat[5].set_visible(False)
    fig.suptitle("Pipelined TPOT (IO overlapped with previous layer compute)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "dsa_tpot_pipelined_vs_bs.png"), dpi=150)
    plt.close(fig)


def plot_io_vs_compute(results):
    """IO time vs compute time per layer — where does IO dominate?"""
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
    fig.suptitle("IO time vs. compute time per layer — can compute hide the IO?",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "dsa_io_vs_compute_bs.png"), dpi=150)
    plt.close(fig)


def plot_offload_gap(results):
    """Offload overhead (offload TPOT / HBM TPOT) vs BS for each seq_len."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, len(SEQ_LENS)))

    for idx, sl in enumerate(SEQ_LENS):
        ratios = []
        for bs in BATCH_SIZES:
            hbm_tpot = results[("hbm", sl, bs)]["tpot_ms"]
            off_tpot = results[("offload", sl, bs)]["tpot_ms"]
            ratios.append(off_tpot / hbm_tpot)
        ax.plot(BATCH_SIZES, ratios, "o-", color=colors[idx], linewidth=2,
                markersize=5, label=f"{sl_label(sl)}")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("batch size")
    ax.set_ylabel("offload slowdown (offload TPOT / HBM TPOT)")
    ax.set_title("Offload overhead vs. batch size by context length")
    ax.set_xscale("log", base=2)
    ax.legend(title="seq_len")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dsa_offload_gap_vs_bs.png"), dpi=150)
    plt.close(fig)


def plot_pipelined_gap(results):
    """Same as offload gap, but with pipelined TPOT."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, len(SEQ_LENS)))

    for idx, sl in enumerate(SEQ_LENS):
        ratios = []
        for bs in BATCH_SIZES:
            hbm_pipe = results[("hbm", sl, bs)]["pipelined_tpot_ms"]
            off_pipe = results[("offload", sl, bs)]["pipelined_tpot_ms"]
            ratios.append(off_pipe / hbm_pipe)
        ax.plot(BATCH_SIZES, ratios, "o-", color=colors[idx], linewidth=2,
                markersize=5, label=f"{sl_label(sl)}")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("batch size")
    ax.set_ylabel("pipelined offload slowdown")
    ax.set_title("Pipelined offload overhead vs. batch size by context length")
    ax.set_xscale("log", base=2)
    ax.legend(title="seq_len")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dsa_pipelined_gap_vs_bs.png"), dpi=150)
    plt.close(fig)


def print_summary(results):
    print("\n" + "=" * 105)
    print(" DSA BATCH-SIZE SWEEP — TPOT comparison (ms)")
    print("=" * 105)
    # Header
    hdr = "%6s  %8s" % ("seq", "mode")
    for bs in BATCH_SIZES:
        hdr += "  BS=%-4d" % bs
    print(hdr)
    print("-" * 105)
    for sl in SEQ_LENS:
        for mode in MODES:
            row = "%6s  %8s" % (sl_label(sl), mode)
            for bs in BATCH_SIZES:
                row += "  %7.1f" % results[(mode, sl, bs)]["tpot_ms"]
            print(row)
        # gap row
        row = "%6s  %8s" % (sl_label(sl), "ratio")
        for bs in BATCH_SIZES:
            h = results[("hbm", sl, bs)]["tpot_ms"]
            o = results[("offload", sl, bs)]["tpot_ms"]
            row += "  %6.2fx" % (o / h)
        print(row)
        print()
    print("-" * 105)

    # Overlap analysis
    print("\n" + "=" * 105)
    print(" OVERLAP ANALYSIS — per-layer IO vs compute (ms), offload mode")
    print(" When compute > IO, the PCIe transfer can be fully pipelined.")
    print("=" * 105)
    hdr = "%6s  %6s" % ("seq", "")
    for bs in BATCH_SIZES:
        hdr += "  BS=%-4d" % bs
    print(hdr)
    print("-" * 105)
    for sl in SEQ_LENS:
        row_io = "%6s  %6s" % (sl_label(sl), "IO")
        row_cmp = "%6s  %6s" % ("", "comp")
        row_flag = "%6s  %6s" % ("", "")
        for bs in BATCH_SIZES:
            r = results[("offload", sl, bs)]
            row_io += "  %7.2f" % r["io_ms"]
            row_cmp += "  %7.2f" % r["compute_ms"]
            flag = "  CMP>" if r["compute_ms"] > r["io_ms"] else "   IO>"
            row_flag += "  %6s " % flag.strip()
        print(row_io)
        print(row_cmp)
        print(row_flag)
        print()
    print("-" * 105)

    # Pipelined TPOT
    print("\n" + "=" * 105)
    print(" PIPELINED TPOT (ms) — if layer IO overlapped with previous layer compute")
    print("=" * 105)
    hdr = "%6s  %8s" % ("seq", "mode")
    for bs in BATCH_SIZES:
        hdr += "  BS=%-4d" % bs
    print(hdr)
    print("-" * 105)
    for sl in SEQ_LENS:
        for mode in MODES:
            row = "%6s  %8s" % (sl_label(sl), mode)
            for bs in BATCH_SIZES:
                row += "  %7.1f" % results[(mode, sl, bs)]["pipelined_tpot_ms"]
            print(row)
        row = "%6s  %8s" % (sl_label(sl), "ratio")
        for bs in BATCH_SIZES:
            h = results[("hbm", sl, bs)]["pipelined_tpot_ms"]
            o = results[("offload", sl, bs)]["pipelined_tpot_ms"]
            row += "  %6.2fx" % (o / h)
        print(row)
        print()
    print("-" * 105)


def main():
    ensure_dir()
    print("Running DSA batch-size sweep...")
    results = run_bs_sweep(SEQ_LENS, BATCH_SIZES, MODES)
    print(f"Computed {len(results)} configurations.")

    print_summary(results)

    print("\nGenerating plots...")
    plot_tpot_vs_bs(results)
    plot_tpot_pipelined(results)
    plot_io_vs_compute(results)
    plot_offload_gap(results)
    plot_pipelined_gap(results)

    print(f"\nPlots written to {FIG_DIR}/:")
    for f in sorted(os.listdir(FIG_DIR)):
        if "bs" in f or "gap" in f or "pipelined" in f:
            print(f"  {f}")


if __name__ == "__main__":
    main()
