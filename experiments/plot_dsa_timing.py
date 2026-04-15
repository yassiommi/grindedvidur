#!/usr/bin/env python3
"""Plots for the DeepSeek-V3 DSA decode timing experiment.

Figures produced (into reports/figures/):
  1. dsa_layer_time_vs_seqlen.png   — per-layer time, both modes
  2. dsa_tpot_vs_seqlen.png         — 61-layer TPOT, both modes
  3. dsa_block_stack_hbm.png        — stacked block A/B/C/MoE breakdown, HBM
  4. dsa_block_stack_offload.png    — same, offload
  5. dsa_io_vs_compute.png          — IO vs compute share by seq_len/mode
  6. dsa_max_batch.png              — max concurrent sequences
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.dsa_layer_analyzer import compute_max_batch, run_sweep

SEQ_LENS = [4096, 32768, 128 * 1024, 512 * 1024, 1024 * 1024]
MODES = ("hbm", "offload")
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reports", "figures")


def sl_label(sl):
    return f"{sl // 1024}K" if sl < 1024 * 1024 else f"{sl // (1024 * 1024)}M"


def ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_layer_time(results):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(SEQ_LENS))
    width = 0.38
    hbm = [results[("hbm", sl)]["layer_total_ms"] for sl in SEQ_LENS]
    off = [results[("offload", sl)]["layer_total_ms"] for sl in SEQ_LENS]

    ax.bar(x - width / 2, hbm, width, label="all-in-memory (HBM)",
           color="#2b7bba")
    ax.bar(x + width / 2, off, width, label="offload (PCIe)",
           color="#e07b39")

    for xi, (h, o) in enumerate(zip(hbm, off)):
        ax.text(xi - width / 2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + width / 2, o, f"{o:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([sl_label(s) for s in SEQ_LENS])
    ax.set_xlabel("sequence length")
    ax.set_ylabel("per-layer decode time (ms)")
    ax.set_title("DeepSeek-V3 DSA: per-layer decode time vs. sequence length")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dsa_layer_time_vs_seqlen.png"), dpi=150)
    plt.close(fig)


def plot_tpot(results):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    hbm = [results[("hbm", sl)]["all_layers_ms"] for sl in SEQ_LENS]
    off = [results[("offload", sl)]["all_layers_ms"] for sl in SEQ_LENS]
    x = np.arange(len(SEQ_LENS))
    ax.plot(x, hbm, "o-", color="#2b7bba", linewidth=2, label="all-in-memory (HBM)")
    ax.plot(x, off, "s-", color="#e07b39", linewidth=2, label="offload (PCIe)")

    for xi, (h, o) in enumerate(zip(hbm, off)):
        ax.annotate(f"{h:.0f}ms", (xi, h), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color="#2b7bba")
        ax.annotate(f"{o:.0f}ms", (xi, o), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color="#e07b39")

    ax.set_xticks(x)
    ax.set_xticklabels([sl_label(s) for s in SEQ_LENS])
    ax.set_xlabel("sequence length")
    ax.set_ylabel("TPOT — time per output token (ms, 61 layers)")
    ax.set_title("DeepSeek-V3 DSA: TPOT vs. sequence length (BS=1)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dsa_tpot_vs_seqlen.png"), dpi=150)
    plt.close(fig)


def plot_block_stack(results, mode: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [sl_label(s) for s in SEQ_LENS]
    max_ab = np.array([results[(mode, s)]["parallel_ab_ms"] for s in SEQ_LENS])
    block_c = np.array([results[(mode, s)]["block_c_ms"] for s in SEQ_LENS])
    moe = np.array([results[(mode, s)]["moe_total_ms"] for s in SEQ_LENS])

    ax.bar(labels, max_ab, label="max(A, B) — Q+index", color="#7bb1df")
    ax.bar(labels, block_c, bottom=max_ab, label="C — attention+output",
           color="#97d0a7")
    ax.bar(labels, moe, bottom=max_ab + block_c, label="MoE",
           color="#e6b866")

    totals = max_ab + block_c + moe
    for xi, t in enumerate(totals):
        ax.text(xi, t, f"{t:.2f}", ha="center", va="bottom", fontsize=8)

    mode_title = "all-in-memory (HBM)" if mode == "hbm" else "offload (PCIe)"
    ax.set_xlabel("sequence length")
    ax.set_ylabel("per-layer time (ms)")
    ax.set_title(f"DSA per-layer composition — {mode_title}")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"dsa_block_stack_{mode}.png"), dpi=150)
    plt.close(fig)


def plot_io_vs_compute(results):
    """Classify each op as IO or compute and show the share by seq_len/mode."""
    io_keys = ["indexer_read_ms", "fetch_kv_ms"]
    # Everything else in the layer is treated as compute-side (kernel work).
    compute_keys_pre = [
        "pre_norm_ms", "q_down_ms", "q_up_ms", "rope_ms",
        "indexer_compute_ms", "topk_ms",
        "kv_up_ms", "attn_core_ms", "o_proj_ms", "residual_ms",
        "moe_norm_ms", "router_gate_ms", "router_softmax_ms", "router_topk_ms",
        "ep_dispatch_ms", "expert_gemm_ms", "ep_combine_ms",
        "shared_expert_ms", "moe_residual_ms",
    ]
    # "compute" here includes HBM-read-bound ops (attn_core, expert_gemm) and
    # NVLink comms; "IO" is the DSA-specific KV/indexer transfers that can be
    # HBM or PCIe depending on mode.

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    width = 0.38
    x = np.arange(len(SEQ_LENS))

    io_hbm = [sum(results[("hbm", s)][k] for k in io_keys) for s in SEQ_LENS]
    io_off = [sum(results[("offload", s)][k] for k in io_keys) for s in SEQ_LENS]
    cmp_hbm = [sum(results[("hbm", s)][k] for k in compute_keys_pre) for s in SEQ_LENS]
    cmp_off = [sum(results[("offload", s)][k] for k in compute_keys_pre) for s in SEQ_LENS]

    ax.bar(x - width / 2, cmp_hbm, width, label="compute+kernel (HBM)",
           color="#2b7bba")
    ax.bar(x - width / 2, io_hbm, width, bottom=cmp_hbm,
           label="KV-IO HBM", color="#9cc3e0")
    ax.bar(x + width / 2, cmp_off, width, label="compute+kernel (offload)",
           color="#e07b39")
    ax.bar(x + width / 2, io_off, width, bottom=cmp_off,
           label="KV-IO PCIe", color="#f5c082")

    ax.set_xticks(x)
    ax.set_xticklabels([sl_label(s) for s in SEQ_LENS])
    ax.set_xlabel("sequence length")
    ax.set_ylabel("per-layer time (ms)")
    ax.set_title("Where the per-layer time goes: compute/kernel vs. KV I/O")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dsa_io_vs_compute.png"), dpi=150)
    plt.close(fig)


def plot_max_batch():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(SEQ_LENS))
    width = 0.38

    hbm_cap = [compute_max_batch("hbm", s)["max_concurrent_seqs"] for s in SEQ_LENS]
    off_cap = [compute_max_batch("offload", s)["max_concurrent_seqs"] for s in SEQ_LENS]

    # Offload mode is "HBM-unbounded" (sentinel -1). We plot it as the
    # capacity ceiling bounded by CPU RAM; for comparison we display the
    # nominal "open" bar using the hbm-free-budget header for context and
    # annotate accordingly.
    hbm_plot = [max(v, 0) for v in hbm_cap]
    off_plot = [max(hbm_plot) * 2 if v < 0 else v for v in off_cap]

    ax.bar(x - width / 2, hbm_plot, width, label="all-in-memory (HBM-bounded)",
           color="#2b7bba")
    ax.bar(x + width / 2, off_plot, width, label="offload (CPU-RAM-bounded)",
           color="#e07b39", alpha=0.7, hatch="//")

    for xi, (h, o) in enumerate(zip(hbm_plot, off_cap)):
        label_h = str(h) if h > 0 else "0 (OOM)"
        label_o = "CPU-bound" if o < 0 else str(o)
        ax.text(xi - width / 2, h, label_h, ha="center", va="bottom", fontsize=8)
        ax.text(xi + width / 2, off_plot[xi], label_o,
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([sl_label(s) for s in SEQ_LENS])
    ax.set_xlabel("sequence length")
    ax.set_ylabel("max concurrent sequences (1× H100 80 GB)")
    ax.set_title("Max concurrent sequences vs. sequence length")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "dsa_max_batch.png"), dpi=150)
    plt.close(fig)


def main():
    ensure_dir()
    results = run_sweep(SEQ_LENS, MODES)
    plot_layer_time(results)
    plot_tpot(results)
    plot_block_stack(results, "hbm")
    plot_block_stack(results, "offload")
    plot_io_vs_compute(results)
    plot_max_batch()
    print(f"Wrote plots to {FIG_DIR}/")
    for f in sorted(os.listdir(FIG_DIR)):
        if f.startswith("dsa_"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
