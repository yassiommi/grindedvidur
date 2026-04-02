#!/usr/bin/env python3
"""Generate all figures for the Layer-Level Timing & KV Prefetch report."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "example_outputs", "experiments", "layer_timing_infersim")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- style ----------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})

C_COMPUTE = "#2ecc71"
C_IO = "#3498db"
C_COMM = "#e67e22"
C_PREFETCH = "#e74c3c"
C_LLAMA = "#9b59b6"
C_DS = "#1abc9c"

# ============================================================
# Data (from experiment outputs)
# ============================================================

# PCIe comparison
pcie = {
    "llama": {
        "pcie4": {"kv": 1.5831, "compute": 0.3206, "prefetch": 0.3106,
                  "e2e": 337.4, "tpot": 0.387, "ratio": 4.94},
        "pcie3": {"kv": 3.1168, "compute": 0.3206, "prefetch": 0.3106,
                  "e2e": 337.2, "tpot": 0.3867, "ratio": 9.72},
    },
    "deepseek": {
        "pcie4": {"kv": 0.3202, "compute": 0.2184, "prefetch": 0.1938,
                  "e2e": 4977.0, "tpot": 9.2567, "ratio": 1.47},
        "pcie3": {"kv": 0.6139, "compute": 0.2183, "prefetch": 0.2052,
                  "e2e": 4578.2, "tpot": 8.5699, "ratio": 2.81},
    },
}

# IO crossover data
io_crossover = {
    "deepseek": {
        "total_decode": 22096, "io_bound_pct": 60.3,
        "kv_range": [0.0872, 0.9246], "compute_range": [0.2041, 0.7601],
        "median_ratio": 1.389, "p90_ratio": 2.393, "p95_ratio": 2.907,
    },
    "llama": {
        "total_decode": 57639, "io_bound_pct": 100.0,
        "kv_range": [1.2407, 4.3869], "compute_range": [0.3119, 1.319],
        "median_ratio": 4.485,
    },
}

# Layer-level breakdown (decode, PCIe4)
llama_layer = {
    "attn_compute": 0.4622, "mlp_compute": 0.6806,
    "kv_load": 1.5831, "tp_comm": 0.0, "prefetch_save": 0.3106,
}
ds_layer = {
    "attn_compute": 0.2077, "mlp_compute": 0.4943,
    "kv_load": 0.3202, "tp_comm": 0.3074, "ep_comm": 0.0,
    "prefetch_save": 0.1938, "routing": 0.0,
}

# ============================================================
# FIGURE 1: Architecture diagram (conceptual)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12); ax.set_ylim(0, 5)
ax.axis("off")
ax.set_title("Framework Architecture: Vidur + InferSim Integration", fontsize=14, fontweight="bold", pad=15)

# Vidur box
rect_v = mpatches.FancyBboxPatch((0.3, 0.5), 4.8, 4.0, boxstyle="round,pad=0.15",
                                   facecolor="#eaf2f8", edgecolor="#2980b9", linewidth=2)
ax.add_patch(rect_v)
ax.text(2.7, 4.15, "Vidur (Event-Driven Simulator)", fontsize=12, fontweight="bold",
        ha="center", color="#2980b9")

# Sub-boxes inside Vidur
for y, label in [(3.3, "Request Scheduler\n(vLLM-style batching)"),
                  (2.2, "Replica Stage Pipeline\n(PP / TP parallelism)"),
                  (1.1, "Execution Time Predictor\n(sklearn + profiling data)")]:
    r = mpatches.FancyBboxPatch((0.7, y-0.35), 4.0, 0.7, boxstyle="round,pad=0.08",
                                 facecolor="white", edgecolor="#5dade2", linewidth=1.2)
    ax.add_patch(r)
    ax.text(2.7, y, label, fontsize=9, ha="center", va="center")

# InferSim box
rect_i = mpatches.FancyBboxPatch((6.5, 0.5), 5.2, 4.0, boxstyle="round,pad=0.15",
                                   facecolor="#eafaf1", edgecolor="#27ae60", linewidth=2)
ax.add_patch(rect_i)
ax.text(9.1, 4.15, "InferSim (Hardware-Aware Modeling)", fontsize=12, fontweight="bold",
        ha="center", color="#27ae60")

for y, label in [(3.3, "FLOPs-Based Compute Model\n(MFU benchmarks per kernel)"),
                  (2.2, "Bandwidth Models\n(PCIe / HBM / NVLink / RDMA)"),
                  (1.1, "MoE Expert Timing\n(routing + grouped GEMM + weight IO)")]:
    r = mpatches.FancyBboxPatch((6.9, y-0.35), 4.4, 0.7, boxstyle="round,pad=0.08",
                                 facecolor="white", edgecolor="#58d68d", linewidth=1.2)
    ax.add_patch(r)
    ax.text(9.1, y, label, fontsize=9, ha="center", va="center")

# Arrow from Vidur to InferSim
ax.annotate("", xy=(6.5, 1.1), xytext=(5.1, 1.1),
            arrowprops=dict(arrowstyle="->", lw=2.5, color="#e74c3c"))
ax.text(5.8, 1.55, "MoE timing\nKV cache IO\nComm model", fontsize=8, ha="center",
        color="#e74c3c", fontstyle="italic")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_architecture.png", bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 2: Three-stream scheduling diagram
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 5), gridspec_kw={"hspace": 0.5})

for idx, (ax, title, has_prefetch) in enumerate([
    (axes[0], "Without KV Prefetch (Sequential I/O)", False),
    (axes[1], "With GPU-Initiated KV Prefetch (Overlapped I/O)", True),
]):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 7.5); ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["NCCL\n(Comm)", "DMA\n(I/O)", "SM\n(Compute)"])
    ax.set_xlabel("Time (ms)")

    if not has_prefetch:
        # Layer N: IO -> Compute -> Comm
        ax.barh(1, 1.2, left=0.0, height=0.6, color=C_IO, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.text(0.6, 1, "KV Load", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.barh(2, 1.0, left=1.2, height=0.6, color=C_COMPUTE, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.text(1.7, 2, "Compute", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.barh(0, 0.5, left=2.2, height=0.6, color=C_COMM, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.text(2.45, 0, "AR", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        # Layer N+1
        ax.barh(1, 1.2, left=2.7, height=0.6, color=C_IO, alpha=0.55, edgecolor="black", linewidth=0.5)
        ax.text(3.3, 1, "KV Load", ha="center", va="center", fontsize=8)
        ax.barh(2, 1.0, left=3.9, height=0.6, color=C_COMPUTE, alpha=0.55, edgecolor="black", linewidth=0.5)
        ax.text(4.4, 2, "Compute", ha="center", va="center", fontsize=8)
        ax.barh(0, 0.5, left=4.9, height=0.6, color=C_COMM, alpha=0.55, edgecolor="black", linewidth=0.5)
        ax.text(5.15, 0, "AR", ha="center", va="center", fontsize=8)
        # Labels
        ax.axvline(x=2.7, color="gray", linestyle="--", alpha=0.5)
        ax.text(1.35, 3.1, "Layer N", ha="center", fontsize=9, fontstyle="italic")
        ax.text(4.05, 3.1, "Layer N+1", ha="center", fontsize=9, fontstyle="italic")
    else:
        # Layer N: Compute + DMA prefetch overlapped
        ax.barh(2, 1.0, left=0.0, height=0.6, color=C_COMPUTE, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.text(0.5, 2, "Compute", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.barh(1, 1.2, left=0.0, height=0.6, color=C_IO, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.text(0.6, 1, "Prefetch N+1", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.barh(0, 0.5, left=1.0, height=0.6, color=C_COMM, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.text(1.25, 0, "AR", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        # Overlap hatching
        ax.barh(1, 1.0, left=0.0, height=0.6, color=C_PREFETCH, alpha=0.25, hatch="///", edgecolor=C_PREFETCH, linewidth=0)
        # Layer N+1
        start_n1 = max(1.2, 1.5)  # max(DMA done, Comm done)
        ax.barh(2, 1.0, left=start_n1, height=0.6, color=C_COMPUTE, alpha=0.55, edgecolor="black", linewidth=0.5)
        ax.text(start_n1+0.5, 2, "Compute", ha="center", va="center", fontsize=8)
        ax.barh(1, 1.2, left=start_n1, height=0.6, color=C_IO, alpha=0.55, edgecolor="black", linewidth=0.5)
        ax.text(start_n1+0.6, 1, "Prefetch N+2", ha="center", va="center", fontsize=8)
        ax.barh(0, 0.5, left=start_n1+1.0, height=0.6, color=C_COMM, alpha=0.55, edgecolor="black", linewidth=0.5)
        ax.text(start_n1+1.25, 0, "AR", ha="center", va="center", fontsize=8)
        ax.axvline(x=start_n1, color="gray", linestyle="--", alpha=0.5)
        ax.text(0.75, 3.1, "Layer N", ha="center", fontsize=9, fontstyle="italic")
        ax.text(start_n1+0.5, 3.1, "Layer N+1", ha="center", fontsize=9, fontstyle="italic")
        ax.annotate("Overlap\nsavings", xy=(0.5, 0.7), fontsize=7, color=C_PREFETCH,
                    ha="center", fontweight="bold")

plt.savefig(f"{OUT_DIR}/fig2_three_stream_scheduling.png", bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 3: Layer breakdown comparison (Llama vs DeepSeek)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, name, data in [
    (axes[0], "Llama-2-7B (Dense, MHA)", {
        "Attention\nCompute": 0.4622, "MLP\nCompute": 0.6806,
        "KV Cache\nLoad": 1.5831, "TP Comm": 0.0, "Prefetch\nSavings": -0.3106}),
    (axes[1], "DeepSeek-V3 (MoE, MLA)", {
        "Attention\nCompute": 0.2077, "MoE Expert\nCompute": 0.4943,
        "KV Cache\nLoad": 0.3202, "TP Comm": 0.3074, "Prefetch\nSavings": -0.1938}),
]:
    labels = list(data.keys())
    values = list(data.values())
    colors = [C_COMPUTE, C_COMPUTE, C_IO, C_COMM, C_PREFETCH]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Time per Layer (ms)")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + (0.03 if v > 0 else -0.06),
                f"{v:.3f}", ha="center", va="bottom" if v > 0 else "top", fontsize=9)

plt.suptitle("Per-Layer Timing Breakdown: Decode Phase (A100, PCIe Gen4)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_layer_breakdown_comparison.png", bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 4: PCIe Gen3 vs Gen4 impact
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 4a: KV Load Time
models = ["Llama-2-7B", "DeepSeek-V3"]
kv_pcie4 = [pcie["llama"]["pcie4"]["kv"], pcie["deepseek"]["pcie4"]["kv"]]
kv_pcie3 = [pcie["llama"]["pcie3"]["kv"], pcie["deepseek"]["pcie3"]["kv"]]
x = np.arange(len(models))
w = 0.35
bars1 = axes[0].bar(x - w/2, kv_pcie4, w, label="PCIe Gen4 (31.5 GB/s)", color="#3498db", edgecolor="black", linewidth=0.5)
bars2 = axes[0].bar(x + w/2, kv_pcie3, w, label="PCIe Gen3 (16 GB/s)", color="#e74c3c", edgecolor="black", linewidth=0.5)
axes[0].set_ylabel("KV Cache Load Time (ms)")
axes[0].set_title("KV Cache Load Time per Layer", fontweight="bold")
axes[0].set_xticks(x); axes[0].set_xticklabels(models)
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{bar.get_height():.2f}", ha="center", fontsize=9)

# 4b: IO/Compute ratio
ratio_pcie4 = [pcie["llama"]["pcie4"]["ratio"], pcie["deepseek"]["pcie4"]["ratio"]]
ratio_pcie3 = [pcie["llama"]["pcie3"]["ratio"], pcie["deepseek"]["pcie3"]["ratio"]]
bars1 = axes[1].bar(x - w/2, ratio_pcie4, w, label="PCIe Gen4", color="#3498db", edgecolor="black", linewidth=0.5)
bars2 = axes[1].bar(x + w/2, ratio_pcie3, w, label="PCIe Gen3", color="#e74c3c", edgecolor="black", linewidth=0.5)
axes[1].set_ylabel("IO / Compute Ratio")
axes[1].set_title("IO-to-Compute Ratio", fontweight="bold")
axes[1].set_xticks(x); axes[1].set_xticklabels(models)
axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Balanced (ratio=1)")
axes[1].legend(fontsize=9)
axes[1].grid(axis="y", alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{bar.get_height():.1f}x", ha="center", fontsize=9)

# 4c: Prefetch savings
save_pcie4 = [pcie["llama"]["pcie4"]["prefetch"], pcie["deepseek"]["pcie4"]["prefetch"]]
save_pcie3 = [pcie["llama"]["pcie3"]["prefetch"], pcie["deepseek"]["pcie3"]["prefetch"]]
bars1 = axes[2].bar(x - w/2, save_pcie4, w, label="PCIe Gen4", color="#3498db", edgecolor="black", linewidth=0.5)
bars2 = axes[2].bar(x + w/2, save_pcie3, w, label="PCIe Gen3", color="#e74c3c", edgecolor="black", linewidth=0.5)
axes[2].set_ylabel("Prefetch Savings (ms/layer)")
axes[2].set_title("KV Prefetch Overlap Savings", fontweight="bold")
axes[2].set_xticks(x); axes[2].set_xticklabels(models)
axes[2].legend(fontsize=9)
axes[2].grid(axis="y", alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9)

plt.suptitle("Impact of PCIe Generation on KV Cache I/O", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_pcie_comparison.png", bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 5: IO-bound percentage pie charts
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, name, io_pct, color in [
    (axes[0], "Llama-2-7B (MHA, TP=1)", 100.0, C_LLAMA),
    (axes[1], "DeepSeek-V3 (MLA, TP=8)", 60.3, C_DS),
]:
    sizes = [io_pct, 100-io_pct]
    labels = [f"IO-Bound\n({io_pct}%)", f"Compute-Bound\n({100-io_pct}%)"]
    colors_pie = [C_IO, C_COMPUTE]
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                       autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
                                       startangle=90, textprops={"fontsize": 10})
    ax.set_title(name, fontsize=11, fontweight="bold")

plt.suptitle("Decode Batch IO-Boundedness (A100, PCIe Gen4)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig5_io_bound_piechart.png", bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 6: KV cache bytes per token comparison (why MLA matters)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

models_kv = ["Llama-2-7B\n(MHA)", "Llama-3-70B\n(GQA)", "DeepSeek-V3\n(MLA)"]
# MHA: 2 * 32 heads * 128 dim * 2 bytes = 16384 bytes
# GQA: 2 * 8 heads * 128 dim * 2 bytes = 4096 bytes
# MLA: (512 + 64) * 2 bytes = 1152 bytes
kv_bytes = [16384, 4096, 1152]
colors_kv = [C_LLAMA, "#f39c12", C_DS]
bars = ax.bar(models_kv, kv_bytes, color=colors_kv, edgecolor="black", linewidth=0.5, alpha=0.85)
ax.set_ylabel("KV Cache per Token per Layer (bytes)")
ax.set_title("KV Cache Size Comparison: Attention Architecture Impact", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, kv_bytes):
    ax.text(bar.get_x() + bar.get_width()/2, v + 200,
            f"{v:,} B\n({v/1024:.1f} KB)", ha="center", fontsize=10, fontweight="bold")
# Add reduction annotations
ax.annotate("4x\nreduction", xy=(1, 4096), xytext=(0.5, 10000),
            fontsize=9, ha="center", color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))
ax.annotate("14.2x\nreduction", xy=(2, 1152), xytext=(1.5, 8000),
            fontsize=9, ha="center", color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig6_kv_cache_size_comparison.png", bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 7: Batch size sweep - why it doesn't matter for sparse models
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 7a: IO/Compute ratio vs batch size (flat line)
batch_sizes = [16, 32, 64, 128, 256, 512]
io_ms = [0.2904] * 6
compute_ms = [0.2164] * 6
ratio_vals = [1.34] * 6

ax = axes[0]
ax.plot(batch_sizes, io_ms, "o-", color=C_IO, linewidth=2, markersize=8, label="Avg KV Load (IO)")
ax.plot(batch_sizes, compute_ms, "s-", color=C_COMPUTE, linewidth=2, markersize=8, label="Avg Compute")
ax.fill_between(batch_sizes, compute_ms, io_ms, alpha=0.15, color=C_IO)
ax.set_xlabel("Batch Size Cap")
ax.set_ylabel("Time per Layer (ms)")
ax.set_title("IO vs Compute: Batch Size Sweep\n(DeepSeek-V3, A100 PCIe Gen4)", fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(batch_sizes)
ax.set_xticklabels([str(b) for b in batch_sizes])
ax.legend()
ax.grid(alpha=0.3)
ax.annotate("IO always exceeds\ncompute regardless\nof batch size", xy=(128, 0.26),
            fontsize=9, ha="center", color="#e74c3c", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaea", edgecolor="#e74c3c"))

# 7b: Conceptual: Context length vs batch size impact on KV IO
ax = axes[1]
ctx_lens = np.array([512, 1024, 2048, 4096, 8192, 16384])
# KV bytes per token for MLA: 1152 bytes
kv_per_token = 1152
bs_examples = [1, 16, 64]
for bs in bs_examples:
    kv_total = kv_per_token * ctx_lens * bs
    kv_load_ms = kv_total / (31.5e9 * 0.8) * 1e3  # PCIe Gen4
    ax.plot(ctx_lens, kv_load_ms, "o-", linewidth=2, markersize=6, label=f"BS={bs}")

# Add compute line (roughly constant)
ax.axhline(y=0.22, color=C_COMPUTE, linestyle="--", linewidth=2, alpha=0.7, label="Compute (constant)")
ax.set_xlabel("Context Length (tokens)")
ax.set_ylabel("KV Load Time per Layer (ms)")
ax.set_title("KV Load Time Scales with Context,\nNot Batch Size (MLA)", fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(ctx_lens)
ax.set_xticklabels([str(c) for c in ctx_lens], fontsize=8)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig7_batch_size_vs_context_length.png", bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 8: Stacked layer timing waterfall
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))

# Simulate 8 layers of DeepSeek-V3 decode with prefetch
n_layers = 8
compute_dur = 0.702  # ms
comm_dur = 0.307  # ms
kv_dur = 0.320  # ms

t = 0
for i in range(n_layers):
    # Compute bar (SM stream)
    ax.barh(i, compute_dur, left=t, height=0.35, color=C_COMPUTE, edgecolor="black", linewidth=0.3)
    # DMA bar (prefetch next layer)
    ax.barh(i-0.4, kv_dur if i < n_layers-1 else 0, left=t, height=0.25, color=C_IO,
            edgecolor="black", linewidth=0.3, alpha=0.7)
    # Overlap hatching
    overlap = min(compute_dur, kv_dur)
    if i < n_layers-1:
        ax.barh(i-0.4, overlap, left=t, height=0.25, color=C_PREFETCH, alpha=0.2, hatch="///", edgecolor=C_PREFETCH, linewidth=0)
    # Comm bar
    ax.barh(i+0.35, comm_dur, left=t+compute_dur, height=0.25, color=C_COMM,
            edgecolor="black", linewidth=0.3, alpha=0.8)

    # Next layer starts at max(DMA done, Comm done)
    t = max(t + kv_dur, t + compute_dur + comm_dur)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Layer Index")
ax.set_yticks(range(n_layers))
ax.set_title("Layer Execution Timeline: DeepSeek-V3 Decode (8 layers, KV Prefetch ON)", fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

legend_elements = [
    mpatches.Patch(facecolor=C_COMPUTE, edgecolor="black", label="SM Compute"),
    mpatches.Patch(facecolor=C_IO, edgecolor="black", label="DMA (KV Prefetch)", alpha=0.7),
    mpatches.Patch(facecolor=C_COMM, edgecolor="black", label="NCCL Comm", alpha=0.8),
    mpatches.Patch(facecolor=C_PREFETCH, edgecolor=C_PREFETCH, label="Overlap Savings", alpha=0.3, hatch="///"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig8_layer_waterfall_deepseek.png", bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 9: Request-level metrics comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# E2E Latency
models_req = ["Llama-2-7B\n(TP=1)", "DeepSeek-V3\n(TP=8, EP=8)"]
e2e = [337.4, 4977.0]
bars = axes[0].bar(models_req, e2e, color=[C_LLAMA, C_DS], edgecolor="black", linewidth=0.5)
axes[0].set_ylabel("Mean E2E Latency (ms)")
axes[0].set_title("End-to-End Latency", fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)
for bar, v in zip(bars, e2e):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 50,
                f"{v:.0f} ms", ha="center", fontsize=10, fontweight="bold")

# TTFT
ttft = [154.0, 254.0]
bars = axes[1].bar(models_req, ttft, color=[C_LLAMA, C_DS], edgecolor="black", linewidth=0.5)
axes[1].set_ylabel("Mean TTFT (ms)")
axes[1].set_title("Time to First Token", fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)
for bar, v in zip(bars, ttft):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 3,
                f"{v:.0f} ms", ha="center", fontsize=10, fontweight="bold")

# TPOT
tpot = [0.387, 9.257]
bars = axes[2].bar(models_req, tpot, color=[C_LLAMA, C_DS], edgecolor="black", linewidth=0.5)
axes[2].set_ylabel("Mean TPOT (ms)")
axes[2].set_title("Time Per Output Token", fontweight="bold")
axes[2].grid(axis="y", alpha=0.3)
for bar, v in zip(bars, tpot):
    axes[2].text(bar.get_x() + bar.get_width()/2, v + 0.1,
                f"{v:.3f} ms", ha="center", fontsize=10, fontweight="bold")

plt.suptitle("Request-Level Performance Metrics (128 requests, QPS=0.5)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig9_request_metrics.png", bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 10: MoE compute model diagram
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 14); ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("MoE Layer Timing Model (InferSim Methodology)", fontsize=14, fontweight="bold", pad=15)

# Input tokens
r = mpatches.FancyBboxPatch((0.5, 4.5), 2.0, 0.8, boxstyle="round,pad=0.1",
                             facecolor="#f0f0f0", edgecolor="black", linewidth=1.5)
ax.add_patch(r); ax.text(1.5, 4.9, "Input\nTokens", ha="center", va="center", fontsize=9, fontweight="bold")

# Router
r = mpatches.FancyBboxPatch((3.5, 4.5), 2.5, 0.8, boxstyle="round,pad=0.1",
                             facecolor="#ffeaa7", edgecolor="#f39c12", linewidth=1.5)
ax.add_patch(r); ax.text(4.75, 4.9, "Router GEMM\nMFU=5%", ha="center", va="center", fontsize=9, fontweight="bold")

# EP Dispatch
r = mpatches.FancyBboxPatch((0.3, 2.8), 2.4, 0.8, boxstyle="round,pad=0.1",
                             facecolor="#fad0c4", edgecolor=C_COMM, linewidth=1.5)
ax.add_patch(r); ax.text(1.5, 3.2, "EP Dispatch\n(NVLink/RDMA)", ha="center", va="center", fontsize=8, fontweight="bold")

# Routed experts
r = mpatches.FancyBboxPatch((3.5, 2.8), 3.0, 0.8, boxstyle="round,pad=0.1",
                             facecolor="#dfe6e9", edgecolor=C_COMPUTE, linewidth=1.5)
ax.add_patch(r); ax.text(5.0, 3.2, "Routed Experts\nGrouped GEMM, MFU=15%", ha="center", va="center", fontsize=8, fontweight="bold")

# Weight loading
r = mpatches.FancyBboxPatch((7.5, 2.8), 2.5, 0.8, boxstyle="round,pad=0.1",
                             facecolor="#dfe6e9", edgecolor=C_IO, linewidth=1.5)
ax.add_patch(r); ax.text(8.75, 3.2, "Expert Weight\nLoading (HBM)", ha="center", va="center", fontsize=8, fontweight="bold")

# max() merge
r = mpatches.FancyBboxPatch((5.5, 1.3), 2.5, 0.7, boxstyle="round,pad=0.1",
                             facecolor="#b8e994", edgecolor="#27ae60", linewidth=1.5)
ax.add_patch(r); ax.text(6.75, 1.65, "max(compute, load)", ha="center", va="center", fontsize=9, fontweight="bold")

# Shared expert
r = mpatches.FancyBboxPatch((9.0, 1.3), 2.5, 0.7, boxstyle="round,pad=0.1",
                             facecolor="#dfe6e9", edgecolor=C_COMPUTE, linewidth=1.5)
ax.add_patch(r); ax.text(10.25, 1.65, "Shared Expert\nDense GEMM, MFU=30%", ha="center", va="center", fontsize=8, fontweight="bold")

# EP Combine
r = mpatches.FancyBboxPatch((9.0, 2.8), 2.5, 0.8, boxstyle="round,pad=0.1",
                             facecolor="#fad0c4", edgecolor=C_COMM, linewidth=1.5)
ax.add_patch(r); ax.text(10.25, 3.2, "EP Combine\n(NVLink/RDMA)", ha="center", va="center", fontsize=8, fontweight="bold")

# Output
r = mpatches.FancyBboxPatch((10.5, 0.2), 2.0, 0.7, boxstyle="round,pad=0.1",
                             facecolor="#f0f0f0", edgecolor="black", linewidth=1.5)
ax.add_patch(r); ax.text(11.5, 0.55, "MoE Output", ha="center", va="center", fontsize=9, fontweight="bold")

# Arrows
for x1, y1, x2, y2 in [
    (2.5, 4.9, 3.5, 4.9),  # Input -> Router
    (4.75, 4.5, 1.5, 3.6),  # Router -> Dispatch
    (4.75, 4.5, 5.0, 3.6),  # Router -> Experts
    (2.7, 3.2, 3.5, 3.2),   # Dispatch -> Experts
    (6.5, 2.8, 6.75, 2.0),  # Experts -> max
    (8.75, 2.8, 6.75, 2.0), # Weight load -> max
    (8.0, 1.65, 9.0, 1.65), # max -> Shared
    (6.5, 3.6, 10.25, 3.6), # Experts -> Combine
    (10.25, 1.3, 11.5, 0.9),# Shared -> Output
    (11.5, 2.8, 11.5, 0.9), # Combine -> Output
]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="gray"))

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig10_moe_timing_model.png", bbox_inches="tight")
plt.close()

# ============================================================
# FIGURE 11: Summary statistics table as figure
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")

col_labels = ["Metric", "Llama-2-7B\n(PCIe4)", "Llama-2-7B\n(PCIe3)", "DeepSeek-V3\n(PCIe4)", "DeepSeek-V3\n(PCIe3)"]
row_data = [
    ["Architecture", "Dense, MHA", "Dense, MHA", "MoE (256E), MLA", "MoE (256E), MLA"],
    ["Layers", "32", "32", "61", "61"],
    ["TP / EP", "1 / 1", "1 / 1", "8 / 8", "8 / 8"],
    ["KV bytes/token/layer", "16,384 B", "16,384 B", "1,152 B", "1,152 B"],
    ["Avg Compute (ms/layer)", "0.321", "0.321", "0.218", "0.218"],
    ["Avg KV Load (ms/layer)", "1.583", "3.117", "0.320", "0.614"],
    ["IO/Compute Ratio", "4.94x", "9.72x", "1.47x", "2.81x"],
    ["Prefetch Savings (ms/layer)", "0.311", "0.311", "0.194", "0.205"],
    ["IO-Bound Batches (%)", "100%", "100%", "60.3%", "~80%"],
    ["Mean E2E Latency (ms)", "337.4", "337.2", "4,977", "4,578"],
    ["Mean TPOT (ms)", "0.387", "0.387", "9.257", "8.570"],
]

table = ax.table(cellText=row_data, colLabels=col_labels, loc="center",
                 cellLoc="center", colColours=["#3498db"]*5)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == 0:
        cell.set_facecolor("#ecf0f1")
        cell.set_text_props(fontweight="bold")
    else:
        cell.set_facecolor("white")
    cell.set_edgecolor("#bdc3c7")

ax.set_title("Complete Experiment Summary", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig11_summary_table.png", bbox_inches="tight")
plt.close()

print(f"All figures saved to {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
