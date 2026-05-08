#!/usr/bin/env python3
"""IndexCache experiment runner.

Sweeps (seq_len, batch_size, mode, F-period, IO-policy) and writes:
  - JSON of all rows
  - A printed summary table
  - Plots:
      1. IO-exposed (=not hidden) heatmap per (mode, F-period)
      2. Speedup (IndexCache pipelined vs. all-F pipelined) heatmap
      3. Compute-vs-IO arithmetic-intensity sweep at fixed seq_len

Run from repo root:
    python -m experiments.run_indexcache_experiment
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from experiments.indexcache_model import simulate, sweep
from experiments.dsa_timing_model import load_profiles, NUM_LAYERS

SEQ_LENS = [4096, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
MODES = ("hbm", "offload")
F_PERIODS = (1, 2, 4, 8)


def sl_label(sl: int) -> str:
    return f"{sl // 1024}K" if sl < 1024 * 1024 else f"{sl // (1024 * 1024)}M"


def print_summary_table(rows):
    """Print concise summary at fp=4 (F:S:S:S)."""
    print(f"\n{'=' * 110}")
    print(" IndexCache (F:S:S:S, pipelined) vs. baseline DSA (all-F, pipelined)")
    print(" IO channel: IDX and KV share one PCIe/HBM bus (their costs sum per layer).")
    print(f"{'=' * 110}")
    hdr = (f"{'sl':>6} {'bs':>4} {'mode':>8} | "
           f"{'compute':>9} {'ic_IO':>9} {'dsa_IO':>9} | "
           f"{'ic_pipe':>9} {'dsa_pipe':>9} {'speedup':>8} | "
           f"{'IO hidden?':>11}")
    print(hdr)
    print("-" * 110)
    for r in rows:
        if r["f_period"] != 4:
            continue
        hidden = "FULL" if r["ic_io_fully_hidden"] else f"{100*r['ic_io_hidden_ms']/max(r['ic_io_total_ms'],1e-9):.0f}%"
        print(f"{sl_label(r['seq_len']):>6} {r['batch_size']:>4} {r['mode']:>8} | "
              f"{r['ic_compute_total_ms']:>9.2f} {r['ic_io_total_ms']:>9.2f} {r['dsa_io_total_ms']:>9.2f} | "
              f"{r['ic_pipe_total_ms']:>9.2f} {r['dsa_pipe_total_ms']:>9.2f} {r['speedup_pipe']:>7.2f}x | "
              f"{hidden:>11}")


def _matrix(rows, mode, f_period, key, seq_lens, batch_sizes):
    by = {(r["seq_len"], r["batch_size"]): r for r in rows
          if r["mode"] == mode and r["f_period"] == f_period}
    M = np.zeros((len(seq_lens), len(batch_sizes)))
    for i, sl in enumerate(seq_lens):
        for j, bs in enumerate(batch_sizes):
            M[i, j] = by[(sl, bs)][key]
    return M


def plot_io_exposed_heatmaps(rows, out_path: str, seq_lens, batch_sizes):
    """Heatmap of exposed IO (ms) — i.e. IO that compute could NOT hide."""
    fig, axes = plt.subplots(len(F_PERIODS), 2, figsize=(11, 3.0 * len(F_PERIODS)),
                             squeeze=False)
    fig.suptitle(f"Exposed IO per decode step (ms) — IO not hidden by compute "
                 f"[{NUM_LAYERS} layers, IDX+KV share IO bus]", fontsize=12)
    vmax = 0.0
    Ms = {}
    for fi, fp in enumerate(F_PERIODS):
        for mi, mode in enumerate(MODES):
            M = _matrix(rows, mode, fp, "ic_io_exposed_ms",
                        seq_lens, batch_sizes)
            Ms[(fi, mi)] = M
            vmax = max(vmax, M.max())
    for fi, fp in enumerate(F_PERIODS):
        for mi, mode in enumerate(MODES):
            ax = axes[fi][mi]
            M = Ms[(fi, mi)]
            im = ax.imshow(M, aspect="auto", cmap="magma_r",
                           norm=plt.matplotlib.colors.LogNorm(vmin=max(M.min(), 1e-2), vmax=max(vmax, 1e-1)),
                           origin="lower")
            label = "DSA (all F)" if fp == 1 else f"IndexCache F:S^{fp-1}"
            ax.set_title(f"{label} | {mode}")
            ax.set_xticks(range(len(batch_sizes)))
            ax.set_xticklabels(batch_sizes)
            ax.set_yticks(range(len(seq_lens)))
            ax.set_yticklabels([sl_label(s) for s in seq_lens])
            for i in range(len(seq_lens)):
                for j in range(len(batch_sizes)):
                    val = M[i, j]
                    txt = "0" if val < 0.05 else f"{val:.0f}" if val >= 1 else f"{val:.1f}"
                    ax.text(j, i, txt, ha="center", va="center",
                            color="white" if val > vmax * 0.3 else "black",
                            fontsize=7)
            if mi == 0:
                ax.set_ylabel("seq_len")
            if fi == len(F_PERIODS) - 1:
                ax.set_xlabel("batch_size")
            plt.colorbar(im, ax=ax)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_speedup_heatmap(rows, out_path: str, seq_lens, batch_sizes,
                         f_period=4):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    fig.suptitle(f"Pipelined IndexCache (F:S^{f_period-1}) speedup over pipelined DSA",
                 fontsize=12)
    vmax = 0.0
    Ms = {}
    for mi, mode in enumerate(MODES):
        M = _matrix(rows, mode, f_period, "speedup_pipe",
                    seq_lens, batch_sizes)
        Ms[mi] = M
        vmax = max(vmax, M.max())
    for mi, mode in enumerate(MODES):
        ax = axes[0][mi]
        M = Ms[mi]
        im = ax.imshow(M, aspect="auto", cmap="viridis",
                       vmin=1.0, vmax=max(vmax, 1.05), origin="lower")
        ax.set_title(f"mode = {mode}")
        ax.set_xticks(range(len(batch_sizes)))
        ax.set_xticklabels(batch_sizes)
        ax.set_yticks(range(len(seq_lens)))
        ax.set_yticklabels([sl_label(s) for s in seq_lens])
        for i in range(len(seq_lens)):
            for j in range(len(batch_sizes)):
                ax.text(j, i, f"{M[i,j]:.2f}x", ha="center", va="center",
                        color="white" if M[i, j] < (1 + vmax) / 2 else "black",
                        fontsize=8)
        ax.set_xlabel("batch_size")
        if mi == 0:
            ax.set_ylabel("seq_len")
        plt.colorbar(im, ax=ax)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_overlap_breakdown(rows, out_path: str, seq_lens, batch_sizes,
                           mode="offload"):
    """For each F-period, show compute, total IO and the IO that compute
    could NOT hide ('exposed' = io_first_cold + sum max(0, io_{i+1} - comp_i)).
    """
    bs_fixed = 16
    fig, axes = plt.subplots(1, len(F_PERIODS), figsize=(4.0 * len(F_PERIODS), 4.0),
                             squeeze=False, sharey=True)
    fig.suptitle(f"Per-step compute vs. IO (ms), {mode}, BS={bs_fixed}, pipelined "
                 f"(IDX+KV share IO bus)", fontsize=12)
    for fi, fp in enumerate(F_PERIODS):
        ax = axes[0][fi]
        comps = []
        ios = []
        exposed = []
        for sl in seq_lens:
            r = next(r for r in rows
                     if r["seq_len"] == sl and r["batch_size"] == bs_fixed
                     and r["mode"] == mode and r["f_period"] == fp)
            comps.append(r["ic_compute_total_ms"])
            ios.append(r["ic_io_total_ms"])
            exposed.append(r["ic_io_exposed_ms"])
        x = np.arange(len(seq_lens))
        ax.bar(x - 0.2, comps, width=0.35, label="compute", color="#4c72b0")
        ax.bar(x + 0.2, ios, width=0.35, label="IO total", color="#dd8452")
        ax.bar(x + 0.2, exposed, width=0.35, label="IO exposed", color="#c44e52")
        label = "DSA (all F)" if fp == 1 else f"F:S^{fp-1}"
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([sl_label(s) for s in seq_lens], rotation=45)
        ax.set_yscale("log")
        if fi == 0:
            ax.set_ylabel("ms (log)")
            ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="reports/figures/indexcache")
    ap.add_argument("--json-out", default="experiments/indexcache_results.json")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)

    print("Running IndexCache sweep...")
    print(f"  seq_lens     = {[sl_label(s) for s in SEQ_LENS]}")
    print(f"  batch_sizes  = {list(BATCH_SIZES)}")
    print(f"  modes        = {list(MODES)}")
    print(f"  F-periods    = {list(F_PERIODS)}  (1=all-F=DSA, 4=F:S:S:S, 8=F:S^7)")

    rows = sweep(SEQ_LENS, BATCH_SIZES, MODES, F_PERIODS)
    print(f"  → {len(rows)} configurations")

    print_summary_table(rows)

    print("\nWriting plots...")
    plot_io_exposed_heatmaps(
        rows, os.path.join(args.out_dir, "fig_io_exposed.png"),
        SEQ_LENS, BATCH_SIZES,
    )
    plot_speedup_heatmap(
        rows, os.path.join(args.out_dir, "fig_speedup_fp4.png"),
        SEQ_LENS, BATCH_SIZES, f_period=4,
    )
    plot_overlap_breakdown(
        rows, os.path.join(args.out_dir, "fig_overlap_offload_bs16.png"),
        SEQ_LENS, BATCH_SIZES, mode="offload",
    )
    plot_overlap_breakdown(
        rows, os.path.join(args.out_dir, "fig_overlap_hbm_bs16.png"),
        SEQ_LENS, BATCH_SIZES, mode="hbm",
    )

    with open(args.json_out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  → wrote {args.json_out}")
    print(f"  → wrote plots in {args.out_dir}/")


if __name__ == "__main__":
    main()
