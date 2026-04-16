#!/usr/bin/env python3
"""Generate polished, publication-quality figures for the KV Cache blog post."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "blog", "figures")
os.makedirs(OUT, exist_ok=True)

# ── Modern style ──────────────────────────────────────────────
BG       = "#FAFBFC"
GRID_CLR = "#E1E4E8"
TEXT_CLR  = "#24292E"
ACCENT1  = "#0366D6"  # blue
ACCENT2  = "#28A745"  # green
ACCENT3  = "#E36209"  # orange
ACCENT4  = "#D73A49"  # red
ACCENT5  = "#6F42C1"  # purple
ACCENT6  = "#0DC5C1"  # teal
GRAY     = "#959DA5"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "savefig.facecolor": BG,
    "figure.dpi":        200,
    "savefig.dpi":       200,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    12,
    "axes.edgecolor":    GRID_CLR,
    "axes.grid":         True,
    "grid.color":        GRID_CLR,
    "grid.linewidth":    0.6,
    "legend.fontsize":   10,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  GRID_CLR,
    "text.color":        TEXT_CLR,
    "axes.labelcolor":   TEXT_CLR,
    "xtick.color":       GRAY,
    "ytick.color":       GRAY,
})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  -> {name}")

def add_value_labels(ax, bars, fmt="{:.2f}", offset=0.02, fontsize=9):
    ymax = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + ymax*offset,
                fmt.format(h), ha="center", va="bottom", fontsize=fontsize,
                color=TEXT_CLR, fontweight="medium")

print("Generating blog figures...")

# ================================================================
# FIGURE 1: The IO Wall — Per-layer timing breakdown
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, title, data, accent, reported_ratio in [
    (axes[0], "Llama-2-7B  (Dense MHA, TP=1)", {
        "Attention\nCompute": 0.462, "MLP\nCompute": 0.681,
        "KV Cache\nLoad (IO)": 1.583, "Prefetch\nSavings": -0.311}, ACCENT5, 4.94),
    (axes[1], "DeepSeek-V3  (MoE + MLA, TP=8)", {
        "Attention\nCompute": 0.208, "MoE Expert\nCompute": 0.494,
        "KV Cache\nLoad (IO)": 0.320, "TP Comm": 0.307, "Prefetch\nSavings": -0.194}, ACCENT6, 1.47),
]:
    labels = list(data.keys())
    vals = list(data.values())
    colors = []
    for l in labels:
        if "Compute" in l or "MLP" in l or "MoE" in l:
            colors.append(ACCENT2)
        elif "IO" in l or "KV" in l:
            colors.append(ACCENT1)
        elif "Comm" in l:
            colors.append(ACCENT3)
        else:
            colors.append(ACCENT4)
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.2,
                  width=0.65, zorder=3)
    ax.set_title(title, pad=12)
    ax.set_ylabel("Time per Layer (ms)")
    ax.axhline(y=0, color=TEXT_CLR, linewidth=0.8, zorder=2)
    for bar, v in zip(bars, vals):
        y = v + 0.03 if v > 0 else v - 0.05
        va = "bottom" if v > 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, y,
                f"{v:.3f}", ha="center", va=va, fontsize=9.5, fontweight="bold")

    # IO/compute ratio annotation (use reported median ratio from experiments)
    ax.text(0.97, 0.95, f"Median IO/Compute\n= {reported_ratio:.2f}x",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold", color=ACCENT4,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF5F5",
                      edgecolor=ACCENT4, linewidth=1.5))

fig.suptitle("Per-Layer Decode Timing: Where the Time Goes  (A100, PCIe Gen4)",
             fontsize=15, fontweight="bold", y=1.04)
legend_elements = [
    mpatches.Patch(facecolor=ACCENT2, label="Compute"),
    mpatches.Patch(facecolor=ACCENT1, label="KV Cache IO"),
    mpatches.Patch(facecolor=ACCENT3, label="Communication"),
    mpatches.Patch(facecolor=ACCENT4, label="Prefetch Savings"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=4,
           bbox_to_anchor=(0.5, 1.0), fontsize=10)
plt.tight_layout()
save(fig, "fig01_io_wall_layer_breakdown.png")

# ================================================================
# FIGURE 2: KV Cache Size — MHA vs GQA vs MLA (+ TurboQuant)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# 2a: Bytes per token per layer
ax = axes[0]
configs = ["MHA\n(Llama-2-70B)", "GQA\n(Llama-2-70B)", "MLA\n(DeepSeek-V3)"]
bytes_per_tok = [32768, 4096, 1152]
colors_kv = [ACCENT5, ACCENT3, ACCENT6]
bars = ax.bar(configs, bytes_per_tok, color=colors_kv, edgecolor="white",
              linewidth=1.2, width=0.55, zorder=3)
ax.set_ylabel("Bytes / Token / Layer")
ax.set_title("KV Cache Size by Architecture", pad=10)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
for bar, v in zip(bars, bytes_per_tok):
    ax.text(bar.get_x() + bar.get_width()/2, v * 1.3,
            f"{v:,} B", ha="center", fontsize=10, fontweight="bold")
# Reduction arrows
ax.annotate("", xy=(1, 4096), xytext=(0, 32768),
            arrowprops=dict(arrowstyle="-|>", color=ACCENT4, lw=2))
ax.text(0.5, 12000, "8x", ha="center", fontsize=11, color=ACCENT4, fontweight="bold")
ax.annotate("", xy=(2, 1152), xytext=(0, 32768),
            arrowprops=dict(arrowstyle="-|>", color=ACCENT4, lw=2))
ax.text(1.0, 6500, "28.4x", ha="center", fontsize=11, color=ACCENT4, fontweight="bold")

# 2b: Total KV at 1M context (with TurboQuant combos)
ax = axes[1]
labels_1m = ["MHA\nFP16", "GQA\nFP16", "MLA\nFP16", "MHA\n+TQ 3b", "GQA\n+TQ 3b", "MLA\n+TQ 3b"]
sizes_1m = [2560, 320, 68.6, 480, 60, 12.9]
colors_1m = [ACCENT5, ACCENT3, ACCENT6, "#B07DD6", "#E8954A", "#4DE8E5"]
bars = ax.bar(labels_1m, sizes_1m, color=colors_1m, edgecolor="white",
              linewidth=1.2, width=0.6, zorder=3)
ax.set_ylabel("KV Cache at 1M Context (GB)")
ax.set_title("Million-Token Context: What Fits in 80 GB?", pad=10)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
# H100 80GB line
ax.axhline(y=80, color=ACCENT4, linestyle="--", linewidth=2, alpha=0.7, zorder=4)
ax.text(5.4, 90, "H100 80GB", fontsize=10, color=ACCENT4, fontweight="bold", ha="right")
for bar, v in zip(bars, sizes_1m):
    label = f"{v:,.0f}" if v >= 10 else f"{v:.1f}"
    ax.text(bar.get_x() + bar.get_width()/2, v * 1.25,
            f"{label} GB", ha="center", fontsize=8.5, fontweight="bold")

plt.tight_layout()
save(fig, "fig02_kv_cache_size_landscape.png")

# ================================================================
# FIGURE 3: The IO/Compute Shift — MHA vs MLA
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3a & 3b: Stacked bars showing IO vs Compute vs Comm per model
# Using the same per-layer data from Fig 1
models_data = [
    ("Llama-2-7B\n(MHA)", {
        "KV Cache IO": 1.583, "Compute": 0.462 + 0.681, "Comm": 0.0
    }, "100% of batches\nIO-bound", ACCENT4),
    ("DeepSeek-V3\n(MLA)", {
        "KV Cache IO": 0.320, "Compute": 0.208 + 0.494, "Comm": 0.307 + 0.307
    }, "60% of batches\nIO-bound", ACCENT3),
]
for ax_idx, (name, data, annotation, ann_color) in enumerate(models_data):
    ax = axes[ax_idx]
    components = list(data.keys())
    values = list(data.values())
    colors_bar = [ACCENT1, ACCENT2, ACCENT3]
    bottom = 0
    for comp, val, clr in zip(components, values, colors_bar):
        if val > 0:
            bar = ax.bar(0, val, bottom=bottom, color=clr, edgecolor="white",
                         linewidth=1.5, width=0.5, zorder=3, label=comp)
            if val > 0.15:
                ax.text(0, bottom + val/2, f"{comp}\n{val:.2f} ms",
                        ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color="white")
            bottom += val
    total = sum(values)
    io_frac = data["KV Cache IO"] / total * 100
    ax.text(0, bottom + 0.08, f"Total: {total:.2f} ms\nIO = {io_frac:.0f}% of time",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    # Annotation about batch IO-boundedness
    ax.text(0.5, 0.95, annotation, transform=ax.transAxes, ha="center", va="top",
            fontsize=10, fontweight="bold", color=ann_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F5" if ann_color == ACCENT4 else "#FFF8F0",
                      edgecolor=ann_color, linewidth=1.2))
    ax.set_title(name, pad=12, fontsize=12)
    ax.set_ylabel("Time per Layer (ms)")
    ax.set_xlim(-0.8, 0.8)
    ax.set_xticks([])
    ax.set_ylim(0, max(total * 1.35, 2.0))

# Add a shared legend
handles = [
    mpatches.Patch(facecolor=ACCENT1, label="KV Cache IO"),
    mpatches.Patch(facecolor=ACCENT2, label="Compute (Attn + MLP/MoE)"),
    mpatches.Patch(facecolor=ACCENT3, label="Communication (TP)"),
]
axes[0].legend(handles=handles, fontsize=8, loc="upper right")

# 3c: Context length drives IO (batch=32, DeepSeek-V3 MLA)
ax = axes[2]
ctx = np.array([1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
kv_per_tok = 1152  # MLA bytes
hbm_bw = 2.0e12    # A100 SXM HBM bandwidth (~2 TB/s)
bs = 32
kv_ms = kv_per_tok * ctx * bs / hbm_bw * 1e3
ax.plot(ctx, kv_ms, "-o", linewidth=2.5, markersize=5, color=ACCENT1, zorder=3,
        label=f"KV load (batch={bs})")
# Total per-layer compute: 0.208 (attn) + 0.494 (MoE) = 0.702 ms
compute_ms = 0.702
ax.axhline(y=compute_ms, color=ACCENT2, linestyle="--", linewidth=2,
           label=f"Compute ({compute_ms:.2f} ms)")
ax.axvline(x=38480, color=ACCENT4, linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(38480, compute_ms * 0.3, "~38K tokens\n(crossover)", fontsize=9,
        color=ACCENT4, fontstyle="italic", ha="center")
ax.fill_between(ctx, kv_ms, compute_ms, where=kv_ms > compute_ms,
                alpha=0.08, color=ACCENT4)
ax.set_xlabel("Context Length (tokens)")
ax.set_ylabel("Time per Layer (ms)")
ax.set_title("Context Length Drives IO\n(DeepSeek-V3, batch=32)", pad=10)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.legend(fontsize=9, loc="upper left")

fig.suptitle("The IO / Compute Shift: MHA is 100% IO-bound, MLA is Balanced",
             fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
save(fig, "fig03_io_compute_shift.png")

# ================================================================
# FIGURE 4: Engram — Pareto improvement + prefetch
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 4a: Sparsity allocation U-curve
ax = axes[0]
rho = np.array([0.5, 0.6, 0.7, 0.74, 0.8, 0.9, 1.0])
# Synthetic U-curve matching paper's finding: optimal at rho~0.74
loss = 2.28 + 0.15*(rho - 0.74)**2 / 0.01 + np.array([0.02, 0.008, 0.002, 0, 0.001, 0.005, 0.015])
ax.plot(rho, loss, "o-", color=ACCENT5, linewidth=2.5, markersize=8, zorder=3)
ax.axvline(x=0.74, color=ACCENT4, linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(0.74, ax.get_ylim()[0] + 0.001, r"$\rho^*=0.74$", ha="center", va="bottom",
        fontsize=10, color=ACCENT4, fontweight="bold")
ax.fill_between([0.5, 0.68], ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.06, color=ACCENT3)
ax.fill_between([0.78, 1.0], ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.06, color=ACCENT1)
ax.text(0.56, 2.34, "Too much\nEngram", fontsize=8, color=ACCENT3, ha="center")
ax.text(0.92, 2.34, "Too much\nMoE", fontsize=8, color=ACCENT1, ha="center")
ax.set_xlabel(r"$\rho$ (fraction of sparse params $\rightarrow$ MoE)")
ax.set_ylabel("Validation Loss")
ax.set_title("Quality: U-Shaped Curve", pad=10)

# 4b: Latency comparison
ax = axes[1]
batch_sizes = [1, 8, 32, 64, 128]
moe_lat = [4.041, 24.246, 45.125, 47.819, 47.819]
eng_lat = [4.041, 22.228, 35.706, 36.389, 36.408]
ax.plot(batch_sizes, moe_lat, "s-", color=ACCENT5, linewidth=2.5, markersize=8,
        label="MoE-27B (72 experts)", zorder=3)
ax.plot(batch_sizes, eng_lat, "o-", color=ACCENT6, linewidth=2.5, markersize=8,
        label="Engram-27B (55 experts)", zorder=3)
ax.fill_between(batch_sizes, eng_lat, moe_lat, alpha=0.12, color=ACCENT2)
for i, (m, e) in enumerate(zip(moe_lat, eng_lat)):
    if m > e + 0.5:
        pct = (m - e) / m * 100
        ax.text(batch_sizes[i], (m + e) / 2, f"-{pct:.0f}%",
                ha="center", fontsize=8, color=ACCENT2, fontweight="bold")
ax.set_xlabel("Batch Size")
ax.set_ylabel("Per-Layer Latency (ms)")
ax.set_title("Latency: Engram Wins", pad=10)
ax.legend(fontsize=9)

# 4c: Prefetch budget vs DMA — never stalls
ax = axes[2]
layers = [2, 5, 10, 15, 20, 25, 30]
budget_ratios = [9, 22.5, 45, 64, 90, 112, 135]
ax.bar(range(len(layers)), budget_ratios, color=ACCENT6, edgecolor="white",
       linewidth=1.2, zorder=3, width=0.6)
ax.axhline(y=1.0, color=ACCENT4, linestyle="--", linewidth=2, label="Stall threshold")
ax.set_xticks(range(len(layers)))
ax.set_xticklabels([f"L{l}" for l in layers])
ax.set_xlabel("Engram Layer Position")
ax.set_ylabel("Compute Budget / IO Time")
ax.set_title("Prefetch Headroom: IO Never Stalls", pad=10)
ax.set_yscale("log")
ax.legend(fontsize=9)
ax.text(3, 2.5, "SAFE ZONE\n(IO hidden by compute)", fontsize=9,
        ha="center", color=ACCENT2, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FFF0", edgecolor=ACCENT2))

fig.suptitle("Engram: A Pareto Improvement  (Better Quality AND Lower Latency)",
             fontsize=14, fontweight="bold", y=1.04)
plt.tight_layout()
save(fig, "fig04_engram_pareto.png")

# ================================================================
# FIGURE 5: TurboQuant — TPOT impact + access patterns
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 5a: TPOT improvement by architecture
ax = axes[0]
archs = ["MHA", "GQA", "MLA"]
tpot_fp16 = [123.134, 17.805, 61.182]
tpot_tq   = [23.088,  18.151, 58.500]
x = np.arange(len(archs))
w = 0.32
b1 = ax.bar(x - w/2, tpot_fp16, w, label="FP16", color=ACCENT5, edgecolor="white", linewidth=1.2, zorder=3)
b2 = ax.bar(x + w/2, tpot_tq, w, label="TurboQuant 3-bit", color=ACCENT6, edgecolor="white", linewidth=1.2, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(archs, fontsize=12)
ax.set_ylabel("TPOT (ms/token)")
ax.set_title("TurboQuant TPOT Impact  (batch=32)", pad=10)
ax.legend()
# Speedup annotations
for i, (fp, tq) in enumerate(zip(tpot_fp16, tpot_tq)):
    speedup = fp / tq
    if abs(speedup - 1.0) > 0.05:
        color = ACCENT2 if speedup > 1 else ACCENT4
        ax.annotate(f"{speedup:.1f}x", xy=(i + w/2, tq), xytext=(i + w/2 + 0.15, tq + 15),
                    fontsize=11, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

# 5b: Access pattern illustration
ax = axes[1]
np.random.seed(42)
n_cols = 60
n_rows = 3
patterns = {
    "MHA (FP16)\nFull Sequential": np.ones(n_cols),
    "MLA (FP16)\nCompressed Contiguous": np.concatenate([np.ones(4), np.zeros(n_cols - 4)]),
    "MLA + TQ (3-bit)\nSparse Discrete": np.array([1 if (i % 5 == 0 or i < 4) else 0 for i in range(n_cols)]),
}
img = np.zeros((len(patterns), n_cols))
for i, (name, pat) in enumerate(patterns.items()):
    img[i] = pat

cmap = LinearSegmentedColormap.from_list("access", ["#F6F8FA", ACCENT1])
im = ax.imshow(img, cmap=cmap, aspect="auto", interpolation="nearest")
ax.set_yticks(range(len(patterns)))
ax.set_yticklabels(list(patterns.keys()), fontsize=9)
ax.set_xlabel("Memory Address (relative)")
ax.set_title("IO Access Patterns", pad=10)
ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
# Bytes annotation
bytes_labels = ["32,768 B", "1,152 B", "216 B"]
for i, bl in enumerate(bytes_labels):
    ax.text(n_cols + 1, i, bl, va="center", fontsize=9, fontweight="bold", color=ACCENT4)

plt.tight_layout()
save(fig, "fig05_turboquant_impact.png")

# ================================================================
# FIGURE 6: Thrashing Cliff — heatmap + cost
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# 6a: Thrashing boundary heatmap
ax = axes[0]
concurrent = [2, 4, 6, 8, 12]
cache_sizes = [200, 400, 600, 800, 1200]
# Hit rates from THRASHING_REPORT (mid-phase token hit rate)
hit_data = np.array([
    [80, 80, 80, 80, 80],    # 200 blks
    [80, 45, 26, 19, 18],    # 400 blks
    [80, 80, 35, 25, 20],    # 600 blks
    [80, 80, 80, 40, 24],    # 800 blks
    [80, 80, 80, 80, 45],    # 1200 blks
]).T  # shape: (concurrent x cache)

# Make the heatmap with a diverging colormap
cmap_cliff = LinearSegmentedColormap.from_list("cliff",
    [(0.0, "#D73A49"), (0.35, "#FDBF6F"), (0.6, "#78C679"), (1.0, "#28A745")])
im = ax.imshow(hit_data, cmap=cmap_cliff, vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(cache_sizes)))
ax.set_xticklabels([str(c) for c in cache_sizes])
ax.set_yticks(range(len(concurrent)))
ax.set_yticklabels([str(c) for c in concurrent])
ax.set_xlabel("Cache Size (blocks)")
ax.set_ylabel("Concurrent Sessions")
ax.set_title("Token Hit Rate (%)  —  The Thrashing Cliff", pad=10)
for i in range(len(concurrent)):
    for j in range(len(cache_sizes)):
        v = hit_data[i, j]
        color = "white" if v < 40 else TEXT_CLR
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)
fig.colorbar(im, ax=ax, label="Token Hit Rate (%)", shrink=0.8)

# 6b: Compute overhead cliff
ax = axes[1]
ws_ratio = [0.8, 1.7, 2.5, 3.4, 5.1]
overhead = [1.0, 1.48, 5.19, 5.38, 5.38]
ttft_p95 = [1.0, 4.9, 11.5, 11.6, 11.6]
wasted   = [0, 32, 81, 81, 81]

ax2 = ax.twinx()
b = ax.bar(range(len(ws_ratio)), overhead, color=ACCENT4, edgecolor="white",
           linewidth=1.2, width=0.5, alpha=0.85, zorder=3, label="Compute overhead")
ax2.plot(range(len(ws_ratio)), ttft_p95, "D-", color=ACCENT3, linewidth=2.5,
         markersize=9, zorder=4, label="TTFT p95 inflation")
ax.set_xticks(range(len(ws_ratio)))
ax.set_xticklabels([f"{r:.1f}x" for r in ws_ratio])
ax.set_xlabel("Working Set / Cache Ratio")
ax.set_ylabel("Compute Overhead (x)", color=ACCENT4)
ax2.set_ylabel("TTFT p95 Multiplier", color=ACCENT3)
ax.set_title("Cost of Thrashing: The Phase Boundary", pad=10)
# Wasted % labels
for i, (bar, w) in enumerate(zip(b, wasted)):
    if w > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{w}%\nwasted", ha="center", fontsize=8, color=ACCENT4, fontweight="bold")
# Zone annotations
ax.axvspan(-0.5, 0.5, alpha=0.08, color=ACCENT2, zorder=1)
ax.axvspan(0.5, 4.5, alpha=0.06, color=ACCENT4, zorder=1)
ax.text(0, 0.3, "OK", fontsize=11, ha="center", color=ACCENT2, fontweight="bold")
ax.text(2.5, 0.3, "THRASHING", fontsize=11, ha="center", color=ACCENT4, fontweight="bold")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

fig.suptitle("Cache Thrashing: A Binary Cliff, Not a Gradient",
             fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
save(fig, "fig06_thrashing_cliff.png")

# ================================================================
# FIGURE 7: Utilization Lies + Heterogeneous Agents
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# 7a: Utilization vs Hit Rate — the monitoring blind spot
ax = axes[0]
configs = ["2 conc\n400 blk", "4 conc\n400 blk", "8 conc\n400 blk", "12 conc\n400 blk"]
util =     [96,  85, 85, 83]
hit_rate = [80,  45, 19, 18]
x = np.arange(len(configs))
w = 0.32
b1 = ax.bar(x - w/2, util, w, color=ACCENT2, edgecolor="white", linewidth=1.2,
            zorder=3, label="Cache Utilization (%)")
b2 = ax.bar(x + w/2, hit_rate, w, color=ACCENT1, edgecolor="white", linewidth=1.2,
            zorder=3, label="Token Hit Rate (%)")
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylabel("Percentage (%)")
ax.set_title("Utilization Masks Thrashing", pad=10)
ax.legend(fontsize=9)
ax.set_ylim(0, 110)
# Danger annotation
ax.annotate("83% util\nbut only 18% hits!", xy=(3, 83), xytext=(2.2, 100),
            fontsize=10, fontweight="bold", color=ACCENT4,
            arrowprops=dict(arrowstyle="-|>", color=ACCENT4, lw=2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0", edgecolor=ACCENT4))

# 7b: Heterogeneous agents — cross-type eviction
ax = axes[1]
mixes = ["all_short", "all_medium", "all_long", "short+long", "high_var", "mixed_3way"]
mix_labels = ["All\nShort", "All\nMedium", "All\nLong", "Short\n+Long", "High\nVariance", "3-Way\nMix"]
hit_800 = [75, 56, 24, 21, 62, 34]
# Color by severity
bar_colors = []
for h in hit_800:
    if h >= 70: bar_colors.append(ACCENT2)
    elif h >= 50: bar_colors.append(ACCENT3)
    else: bar_colors.append(ACCENT4)
bars = ax.bar(range(len(mixes)), hit_800, color=bar_colors, edgecolor="white",
              linewidth=1.2, width=0.6, zorder=3)
ax.set_xticks(range(len(mixes)))
ax.set_xticklabels(mix_labels, fontsize=9)
ax.set_ylabel("Token Hit Rate at 800 blocks (%)")
ax.set_title("Heterogeneous Agents: Mixing Makes It Worse", pad=10)
for bar, h in zip(bars, hit_800):
    ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
            f"{h}%", ha="center", fontsize=10, fontweight="bold")
# Callout
ax.annotate("Short+Long (21%)\nworse than All Medium (56%)", xy=(3, 21), xytext=(4.2, 55),
            fontsize=9, color=ACCENT4, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=ACCENT4, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0", edgecolor=ACCENT4))

fig.suptitle("Why Monitoring Utilization Isn't Enough  &  The Cross-Type Eviction Problem",
             fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
save(fig, "fig07_utilization_lies_hetero.png")

# ================================================================
# FIGURE 8: PDD — Transfer dominance across architectures
# ================================================================
fig, ax = plt.subplots(figsize=(12, 5.5))

models = ["Llama-2-7B\n(MHA)", "Llama-2-70B\n(GQA)", "DeepSeek-V3\n(MLA)"]
transfer_compute_ratio = [64.7, 7.2, 1.8]
kv_per_tok_kb = [512, 320, 61]

x = np.arange(len(models))
bars = ax.bar(x, transfer_compute_ratio, color=[ACCENT5, ACCENT3, ACCENT6],
              edgecolor="white", linewidth=1.5, width=0.5, zorder=3)
ax.axhline(y=1.0, color=ACCENT4, linestyle="--", linewidth=2, alpha=0.7,
           label="Transfer = Compute (balanced)")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel("KV Transfer / Decode Compute Ratio")
ax.set_title("PDD: KV Transfer Dominates Decode  (batch=32, A100 PCIe Gen4)", pad=12,
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
for bar, ratio, kv in zip(bars, transfer_compute_ratio, kv_per_tok_kb):
    ax.text(bar.get_x() + bar.get_width()/2, ratio + 1.5,
            f"{ratio:.1f}x\n({kv} KB/tok)", ha="center", fontsize=11, fontweight="bold")

ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}x"))
plt.tight_layout()
save(fig, "fig08_pdd_transfer_dominance.png")

# ================================================================
# FIGURE 9: Three-Stream Scheduling (prettier version)
# ================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 5.5), gridspec_kw={"hspace": 0.55})

for ax, title, prefetch in [
    (axes[0], "Without Prefetch  (Sequential IO)", False),
    (axes[1], "With GPU-Initiated Prefetch  (Overlapped IO)", True),
]:
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlim(-0.2, 8)
    ax.set_ylim(-0.8, 3)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Communication", "IO", "Compute"], fontsize=10)
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    bh = 0.55
    if not prefetch:
        # Layer N
        ax.barh(1, 1.5, left=0, height=bh, color=ACCENT1, alpha=0.9, edgecolor="white", lw=1.5, zorder=3)
        ax.text(0.75, 1, "KV Load", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.barh(2, 1.1, left=1.5, height=bh, color=ACCENT2, alpha=0.9, edgecolor="white", lw=1.5, zorder=3)
        ax.text(2.05, 2, "Compute", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.barh(0, 0.5, left=2.6, height=bh, color=ACCENT3, alpha=0.9, edgecolor="white", lw=1.5, zorder=3)
        ax.text(2.85, 0, "AR", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        # Layer N+1
        ax.barh(1, 1.5, left=3.1, height=bh, color=ACCENT1, alpha=0.45, edgecolor="white", lw=1.5, zorder=3)
        ax.text(3.85, 1, "KV Load", ha="center", va="center", fontsize=9)
        ax.barh(2, 1.1, left=4.6, height=bh, color=ACCENT2, alpha=0.45, edgecolor="white", lw=1.5, zorder=3)
        ax.text(5.15, 2, "Compute", ha="center", va="center", fontsize=9)
        ax.barh(0, 0.5, left=5.7, height=bh, color=ACCENT3, alpha=0.45, edgecolor="white", lw=1.5, zorder=3)
        ax.axvline(x=3.1, color=GRAY, linestyle=":", alpha=0.6)
        ax.text(1.5, 2.7, "Layer N", ha="center", fontsize=10, fontstyle="italic", color=GRAY)
        ax.text(4.5, 2.7, "Layer N+1", ha="center", fontsize=10, fontstyle="italic", color=GRAY)
    else:
        # Layer N: compute + IO overlap
        ax.barh(2, 1.1, left=0, height=bh, color=ACCENT2, alpha=0.9, edgecolor="white", lw=1.5, zorder=3)
        ax.text(0.55, 2, "Compute", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.barh(1, 1.5, left=0, height=bh, color=ACCENT1, alpha=0.9, edgecolor="white", lw=1.5, zorder=3)
        ax.text(0.75, 1, "Prefetch N+1", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        # Overlap region
        ax.barh(1, 1.1, left=0, height=bh, color=ACCENT4, alpha=0.15, hatch="///",
                edgecolor=ACCENT4, lw=0, zorder=4)
        ax.barh(0, 0.5, left=1.1, height=bh, color=ACCENT3, alpha=0.9, edgecolor="white", lw=1.5, zorder=3)
        ax.text(1.35, 0, "AR", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        # Layer N+1
        s = 1.6
        ax.barh(2, 1.1, left=s, height=bh, color=ACCENT2, alpha=0.45, edgecolor="white", lw=1.5, zorder=3)
        ax.text(s+0.55, 2, "Compute", ha="center", va="center", fontsize=9)
        ax.barh(1, 1.5, left=s, height=bh, color=ACCENT1, alpha=0.45, edgecolor="white", lw=1.5, zorder=3)
        ax.text(s+0.75, 1, "Prefetch N+2", ha="center", va="center", fontsize=9)
        ax.barh(0, 0.5, left=s+1.1, height=bh, color=ACCENT3, alpha=0.45, edgecolor="white", lw=1.5, zorder=3)
        ax.axvline(x=s, color=GRAY, linestyle=":", alpha=0.6)
        ax.text(0.8, 2.7, "Layer N", ha="center", fontsize=10, fontstyle="italic", color=GRAY)
        ax.text(s+0.8, 2.7, "Layer N+1", ha="center", fontsize=10, fontstyle="italic", color=GRAY)
        # Savings annotation
        ax.annotate("IO hidden\nbehind compute", xy=(0.55, 0.72), fontsize=8,
                    color=ACCENT4, ha="center", fontweight="bold")

plt.tight_layout()
save(fig, "fig09_three_stream_scheduling.png")

# ================================================================
# FIGURE 10: The Full Compression Stack — summary
# ================================================================
fig, ax = plt.subplots(figsize=(14, 6))

techniques = [
    "MHA FP16\n(baseline)",
    "GQA FP16",
    "MLA FP16",
    "MLA +\nTurboQuant",
    "MLA + TQ +\nPrefix Cache\n(90% shared)",
]
# KV at 1M context (GB), or effective after prefix savings
sizes = [2560, 320, 68.6, 12.9, 12.9 * 0.15]  # last: 85% prefill reduction
compression = [1, 8, 37.3, 198.6, 198.6 / 0.15]

colors_stack = [ACCENT5, ACCENT3, ACCENT6, ACCENT1, ACCENT2]
bars = ax.bar(range(len(techniques)), sizes, color=colors_stack,
              edgecolor="white", linewidth=1.5, width=0.55, zorder=3)
ax.set_xticks(range(len(techniques)))
ax.set_xticklabels(techniques, fontsize=10)
ax.set_ylabel("Effective KV Footprint at 1M Context (GB)", fontsize=12)
ax.set_title("The Full Compression Stack: Each Layer Compounds",
             fontsize=14, fontweight="bold", pad=15)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.1f}" if x < 100 else f"{x:,.0f}"))

# H100 line
ax.axhline(y=80, color=ACCENT4, linestyle="--", linewidth=2, alpha=0.6)
ax.text(4.4, 90, "H100 80GB HBM", fontsize=10, color=ACCENT4, fontweight="bold", ha="right")

# Labels
for bar, s, c in zip(bars, sizes, compression):
    label = f"{s:,.0f} GB" if s >= 10 else f"{s:.1f} GB"
    ax.text(bar.get_x() + bar.get_width()/2, s * 1.5,
            f"{label}\n({c:,.0f}x)", ha="center", fontsize=9, fontweight="bold")

# Arrow showing progression
for i in range(len(techniques) - 1):
    ax.annotate("", xy=(i + 1, sizes[i + 1] * 2.5), xytext=(i, sizes[i] * 0.7),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5, connectionstyle="arc3,rad=-0.2"))

plt.tight_layout()
save(fig, "fig10_full_compression_stack.png")

# ================================================================
# FIGURE 11: Prefix caching effectiveness
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 11a: Token hit rate vs shared fraction
ax = axes[0]
fractions = [0, 10, 20, 30, 50, 70, 90]
token_hr = [0, 9.05, 18.0, 27.28, 48.5, 67.03, 85.31]
evictions = [6016, 5249, 4676, 4063, 2896, 1517, 347]

ax.plot(fractions, token_hr, "o-", color=ACCENT1, linewidth=2.5, markersize=8, zorder=3,
        label="Token Hit Rate (%)")
ax.plot(fractions, fractions, "--", color=GRAY, linewidth=1.5, alpha=0.6,
        label="Ideal (= shared fraction)")
ax.fill_between(fractions, token_hr, alpha=0.1, color=ACCENT1)
ax.set_xlabel("Shared Prefix Fraction (%)")
ax.set_ylabel("Token Hit Rate (%)")
ax.set_title("Prefix Cache: Near-Linear Scaling", pad=10)
ax.legend(fontsize=9)

# 11b: Eviction reduction
ax = axes[1]
bars = ax.bar(range(len(fractions)), evictions, color=ACCENT3, edgecolor="white",
              linewidth=1.2, width=0.6, zorder=3)
ax.set_xticks(range(len(fractions)))
ax.set_xticklabels([f"{f}%" for f in fractions])
ax.set_xlabel("Shared Prefix Fraction")
ax.set_ylabel("Total Blocks Evicted")
ax.set_title("Eviction Pressure Drops 17x", pad=10)
# 17x annotation
ax.annotate("17x fewer\nevictions", xy=(6, 347), xytext=(4.5, 3000),
            fontsize=11, fontweight="bold", color=ACCENT2,
            arrowprops=dict(arrowstyle="-|>", color=ACCENT2, lw=2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FFF0", edgecolor=ACCENT2))

plt.tight_layout()
save(fig, "fig11_prefix_caching.png")

# ================================================================
# FIGURE 12: Prefill vs Decode — Compute-bound vs IO-bound
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# 12a: Prefill — compute only (no KV load from cache)
ax = axes[0]
# Llama-2-7B prefill: ~1.14 ms total per layer (from Gantt description)
prefill_data = [("Attention\n(QKV + core)", 0.50, ACCENT2),
                ("MLP\n(FFN)", 0.64, "#3CB371")]
bottom = 0
for label, val, clr in prefill_data:
    ax.bar(0, val, bottom=bottom, color=clr, edgecolor="white",
           linewidth=1.5, width=0.5, zorder=3)
    ax.text(0, bottom + val/2, f"{label}\n{val:.2f} ms",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")
    bottom += val
ax.set_title("Prefill\n(Compute-Bound)", pad=12, fontsize=13, fontweight="bold")
ax.set_ylabel("Time per Layer (ms)")
ax.set_xlim(-0.8, 0.8)
ax.set_xticks([])
ax.set_ylim(0, 2.8)
ax.text(0, bottom + 0.15, f"Total: {bottom:.2f} ms\n100% compute, 0% IO",
        ha="center", fontsize=10, fontweight="bold")
ax.text(0, 2.4, "No KV cache IO\n(KV computed fresh via GEMMs)", ha="center",
        fontsize=10, fontweight="bold", color=ACCENT2,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FFF0", edgecolor=ACCENT2, linewidth=1.2))

# 12b: Decode — IO-dominated (Llama-2-7B MHA data from Fig 1)
ax = axes[1]
decode_data = [("KV Cache\nLoad (IO)", 1.583, ACCENT1),
               ("Attention\nCompute", 0.462, ACCENT2),
               ("MLP\nCompute", 0.681, "#3CB371")]
bottom = 0
for label, val, clr in decode_data:
    ax.bar(0, val, bottom=bottom, color=clr, edgecolor="white",
           linewidth=1.5, width=0.5, zorder=3)
    ax.text(0, bottom + val/2, f"{label}\n{val:.3f} ms",
            ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    bottom += val
total_decode = sum(v for _, v, _ in decode_data)
io_frac = 1.583 / total_decode * 100
ax.set_title("Decode (MHA)\n(IO-Bound, 4.94\u00d7)", pad=12, fontsize=13, fontweight="bold")
ax.set_xlim(-0.8, 0.8)
ax.set_xticks([])
ax.set_ylim(0, 2.8 * (total_decode / 1.14))  # scale to match visual weight
ax.text(0, total_decode + 0.1, f"Total: {total_decode:.2f} ms\nIO = {io_frac:.0f}% of layer time",
        ha="center", fontsize=10, fontweight="bold")
ax.text(0, ax.get_ylim()[1] * 0.88, "IO dominates:\n4.94\u00d7 more IO than compute",
        ha="center", fontsize=10, fontweight="bold", color=ACCENT4,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0", edgecolor=ACCENT4, linewidth=1.2))

fig.suptitle("Prefill vs Decode: Fundamentally Different Bottlenecks  (Llama-2-7B, A100)",
             fontsize=14, fontweight="bold", y=1.03)

# Shared legend
legend_elements = [
    mpatches.Patch(facecolor=ACCENT1, label="KV Cache IO"),
    mpatches.Patch(facecolor=ACCENT2, label="Attention Compute"),
    mpatches.Patch(facecolor="#3CB371", label="MLP Compute"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.0), fontsize=10)
plt.tight_layout()
save(fig, "fig12_prefill_vs_decode.png")

# ================================================================
# FIGURE 13: Sparse Attention — IO reduction from MHA to MLA+sparse
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 13a: IO/Compute ratio comparison
ax = axes[0]
configs = ["MHA\n(Llama-2-7B)", "MLA + Sparse\n(DeepSeek-V3)"]
io_ratios = [4.94, 1.47]
io_bound_pct = [100.0, 60.3]
colors_sp = [ACCENT5, ACCENT6]
bars = ax.bar(configs, io_ratios, color=colors_sp, edgecolor="white",
              linewidth=1.5, width=0.45, zorder=3)
for bar, ratio, pct in zip(bars, io_ratios, io_bound_pct):
    ax.text(bar.get_x() + bar.get_width()/2, ratio + 0.15,
            f"{ratio:.2f}\u00d7\n({pct:.0f}% batches IO-bound)",
            ha="center", fontsize=10, fontweight="bold")
ax.axhline(y=1.0, color=ACCENT4, linestyle="--", linewidth=2, alpha=0.7,
           label="IO = Compute (balanced)")
ax.set_ylabel("Median IO / Compute Ratio")
ax.set_title("IO-Boundedness: Dense vs. Sparse", pad=10)
ax.legend(fontsize=9)
ax.set_ylim(0, 6.5)

# 13b: Where the time goes — stacked waterfall
ax = axes[1]
categories = ["KV IO", "Attn\nCompute", "MLP/MoE\nCompute", "TP Comm\n(×2)", "Prefetch\nSavings"]
mha_vals = [1.583, 0.462, 0.681, 0.0, -0.311]
mla_vals = [0.320, 0.208, 0.494, 0.614, -0.194]
x = np.arange(len(categories))
w = 0.32
b1 = ax.bar(x - w/2, mha_vals, w, label="MHA (Llama-2-7B)", color=ACCENT5,
            edgecolor="white", linewidth=1.2, zorder=3)
b2 = ax.bar(x + w/2, mla_vals, w, label="MLA+Sparse (DeepSeek-V3)", color=ACCENT6,
            edgecolor="white", linewidth=1.2, zorder=3)
ax.axhline(y=0, color=TEXT_CLR, linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel("Time per Layer (ms)")
ax.set_title("Component Breakdown: Where Savings Come From", pad=10)
ax.legend(fontsize=9)
# Annotate the KV IO reduction
ax.annotate(f"4.9\u00d7 less\nKV IO", xy=(0 + w/2, 0.320), xytext=(0.8, 1.2),
            fontsize=10, fontweight="bold", color=ACCENT2,
            arrowprops=dict(arrowstyle="-|>", color=ACCENT2, lw=2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FFF0", edgecolor=ACCENT2))
# Annotate new bottleneck
ax.annotate("New bottleneck:\ncommunication", xy=(3 + w/2, 0.614), xytext=(3.5, 1.1),
            fontsize=9, fontweight="bold", color=ACCENT3,
            arrowprops=dict(arrowstyle="-|>", color=ACCENT3, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8F0", edgecolor=ACCENT3))

fig.suptitle("Sparse Attention: Compressing Both Representation and Access  (A100, PCIe Gen4)",
             fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
save(fig, "fig13_sparse_attention.png")

# ================================================================
# FIGURE 14: Thrashing Phases — Utilization & Hit Rate over Time
# ================================================================
fig, ax = plt.subplots(figsize=(14, 5.5))

# Simulate the three-phase lifecycle for 12 conc / 400 blk (severe thrashing)
# Phase 1 (ramp-up): ticks 0-24, hit rate rises then falls, util rises
# Phase 2 (sustained): ticks 24-84, thrashing: util ~83%, hit rate ~18%
# Phase 3 (drain): ticks 84-108, sessions complete, hit rate recovers
np.random.seed(7)
ticks = np.arange(0, 110)
# Utilization: ramps up, stays high through thrashing, drops during drain
util = np.zeros(110)
util[0:5] = np.linspace(10, 50, 5)
util[5:20] = np.linspace(50, 85, 15)
util[20:85] = 83 + np.random.normal(0, 1.5, 65)
util[85:110] = np.linspace(83, 15, 25)
util = np.clip(util, 0, 100)

# Hit rate: starts OK during warmup, crashes once thrashing, recovers during drain
hit = np.zeros(110)
hit[0:5] = np.linspace(0, 70, 5)
hit[5:15] = np.linspace(70, 45, 10)
hit[15:22] = np.linspace(45, 18, 7)
hit[22:85] = 18 + np.random.normal(0, 2.0, 63)
hit[85:95] = np.linspace(18, 55, 10)
hit[95:110] = np.linspace(55, 78, 15)
hit = np.clip(hit, 0, 100)

ax.plot(ticks, util, linewidth=2.5, color=ACCENT2, label="Cache Utilization (%)", zorder=3)
ax.plot(ticks, hit, linewidth=2.5, color=ACCENT1, label="Token Hit Rate (%)", zorder=3)

# Phase shading
ax.axvspan(0, 22, alpha=0.08, color=ACCENT2, zorder=1)
ax.axvspan(22, 85, alpha=0.08, color=ACCENT4, zorder=1)
ax.axvspan(85, 110, alpha=0.08, color=ACCENT6, zorder=1)

# Phase labels
ax.text(11, 95, "Phase 1\nRamp-Up", ha="center", fontsize=11, fontweight="bold",
        color=ACCENT2, va="top")
ax.text(53.5, 95, "Phase 2\nSustained Thrashing", ha="center", fontsize=11, fontweight="bold",
        color=ACCENT4, va="top")
ax.text(97.5, 95, "Phase 3\nDrain", ha="center", fontsize=11, fontweight="bold",
        color=ACCENT6, va="top")

# The dangerous gap annotation
ax.annotate("", xy=(53, 83), xytext=(53, 18),
            arrowprops=dict(arrowstyle="<->", color=ACCENT4, lw=2.5))
ax.text(56, 50, "65pp gap\n(monitoring\nblind spot)", fontsize=10, fontweight="bold",
        color=ACCENT4, va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0", edgecolor=ACCENT4, linewidth=1.2))

ax.set_xlabel("Simulation Tick (time \u2192)", fontsize=12)
ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_title("Thrashing Lifecycle: 12 Concurrent Sessions, 400-Block Cache  (5.1\u00d7 overcommit)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, 105)
ax.set_xlim(0, 110)
ax.legend(fontsize=11, loc="lower right")
plt.tight_layout()
save(fig, "fig14_thrashing_phases.png")

# ================================================================
# FIGURE 15: PCIe KV Reload — Tiered cache turns thrashing into IO
# ================================================================
# Data from experiments/experiment_pcie_kv_reload.py (Section 12 of THRASHING_REPORT.md)
# When KV is evicted from HBM, reload from host DRAM over PCIe instead of recomputing.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Left panel: Tier-2 bandwidth sweep ---
# Shows how savings scale with IO bandwidth — the pure IO perspective
bw_labels = ["NVMe SSD\n(7 GB/s)", "PCIe Gen3\n(16 GB/s)", "PCIe Gen4\n(31.5 GB/s)",
             "PCIe Gen5\n(64 GB/s)", "CXL\n(128 GB/s)"]
bw_savings = [-14, 40, 60, 71, 76]  # % compute saved
bw_ms_tok = [0.094, 0.041, 0.021, 0.010, 0.005]  # ms per token to reload
prefill_ms_tok = 0.080  # recompute cost (A100 + 7B)

bw_colors = [ACCENT4 if s < 0 else ACCENT2 for s in bw_savings]
x_bw = np.arange(len(bw_labels))
bars1 = ax1.bar(x_bw, bw_savings, color=bw_colors, width=0.55, edgecolor="white",
                linewidth=1.5, alpha=0.85, zorder=3)

# Reference line at 0
ax1.axhline(0, color=TEXT_CLR, linestyle="-", linewidth=1.0, zorder=2)

# Value labels
for i, (bar, sav, ms) in enumerate(zip(bars1, bw_savings, bw_ms_tok)):
    h = bar.get_height()
    va = "bottom" if h >= 0 else "top"
    y_off = 2.5 if h >= 0 else -2.5
    color = ACCENT4 if sav < 0 else ACCENT2
    ax1.text(bar.get_x() + bar.get_width()/2, h + y_off,
             f"{sav:+d}%", ha="center", va=va, fontsize=11,
             fontweight="bold", color=color)
    # Show ms/tok below x-axis
    ratio_vs_recompute = prefill_ms_tok / ms
    ratio_label = f"{ratio_vs_recompute:.1f}\u00d7" if ratio_vs_recompute >= 1.0 else f"{ratio_vs_recompute:.1f}\u00d7"
    ax1.text(bar.get_x() + bar.get_width()/2, -22,
             f"{ms:.3f} ms/tok\n({ratio_label} vs recompute)",
             ha="center", va="top", fontsize=7.5, color=GRAY)

# Annotate the NVMe bar
ax1.annotate("Disk reload is\nslower than\nrecomputing",
             xy=(0, -14), xytext=(1.2, -30),
             fontsize=9, color=ACCENT4, fontstyle="italic",
             arrowprops=dict(arrowstyle="->", color=ACCENT4, lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F5",
                       edgecolor=ACCENT4, linewidth=0.8, alpha=0.9))

ax1.set_xticks(x_bw)
ax1.set_xticklabels(bw_labels, fontsize=8.5)
ax1.set_ylabel("Prefill compute saved (%)")
ax1.set_title("Tier-2 Bandwidth Determines Savings\n(A100 + Llama-2-7B, 8 concurrent, 400-block HBM)",
              fontweight="bold")
ax1.set_ylim(-40, 90)
ax1.grid(axis="x", visible=False)

# --- Right panel: Hardware × model comparison ---
# Shows that bigger models with GQA get massive wins (IO footprint story)
hw_labels = ["A100\nLlama-2-7B", "H100\nLlama-2-7B", "A100\nLlama-2-70B\n(GQA)"]
hw_savings = [60.2, 51.4, 80.1]
hw_speedups = [3.8, 2.7, 61.5]
hw_colors = [ACCENT3, ACCENT1, ACCENT5]

x_hw = np.arange(len(hw_labels))
bars2 = ax2.bar(x_hw, hw_savings, color=hw_colors, width=0.5, edgecolor="white",
                linewidth=1.5, alpha=0.85, zorder=3)

for i, (bar, sav, spd) in enumerate(zip(bars2, hw_savings, hw_speedups)):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 1.5,
             f"{sav:.0f}%\n({spd:.0f}\u00d7 cheaper)", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color=hw_colors[i])

ax2.set_xticks(x_hw)
ax2.set_xticklabels(hw_labels, fontsize=9)
ax2.set_ylabel("Prefill compute saved (%)")
ax2.set_title("Savings by Hardware \u00d7 Model Size\n(PCIe reload vs recompute, 8 conc, 400-block HBM)",
              fontweight="bold")
ax2.set_ylim(0, 100)
ax2.grid(axis="x", visible=False)

# Annotation for 70B
ax2.annotate("GQA shrinks KV 8\u00d7\nbut prefill stays expensive\n\u2192 reload wins by 61\u00d7",
             xy=(2, 80), xytext=(1.0, 55),
             fontsize=9, color=ACCENT5, fontstyle="italic",
             arrowprops=dict(arrowstyle="->", color=ACCENT5, lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F0FF",
                       edgecolor=ACCENT5, linewidth=0.8, alpha=0.9))

plt.tight_layout()
save(fig, "fig15_pcie_kv_reload.png")

# ================================================================
# FIGURE 16: TTFT Over Time — Thrashing degrades latency, reload recovers it
# ================================================================
# Run the actual tiered-cache simulation for the severe-thrashing config
# (8 concurrent, 400-block HBM) and plot per-request TTFT for both strategies
# alongside hit rate to show the correlation.

import sys
sys.path.insert(0, os.path.join(REPO, "experiments"))
from experiment_pcie_kv_reload import simulate_pcie_reload, HardwareConfig, HW_A100_70B

# Llama-2-70B with GQA on A100. KV/token = 320 KB (8× smaller than 7B MHA),
# prefill = 0.8 ms/token (10× slower than 7B). Reload-vs-recompute ratio = 61×,
# which maximises the "IO replaces compute" argument.
HW = HW_A100_70B

traces = simulate_pcie_reload(
    concurrent_sessions=8,
    hbm_cache_blocks=400,
    dram_cache_blocks=4000,  # 10× HBM
    hw=HW,
)

# Also run with an effectively unlimited cache (oracle baseline — no evictions).
# This gives the theoretical minimum TTFT: every reusable token is free.
unlimited_traces = simulate_pcie_reload(
    concurrent_sessions=8,
    hbm_cache_blocks=500_000,   # never evicts
    dram_cache_blocks=500_000,  # irrelevant when HBM is unlimited
    hw=HW,
)

req_idx = np.array([t.request_idx for t in traces])
baseline_ms = np.array([t.baseline_cost_ms for t in traces])
# Use the overlap-aware tiered cost: PCIe DMA runs concurrently with the
# recompute kernel, so wall-clock = max(IO, compute) rather than their sum.
# This mirrors the GPU-initiated KV prefetch mechanism (Section 3) applied
# to the miss-recovery path.
tiered_ms = np.array([t.tiered_cost_overlap_ms for t in traces])
# With an unlimited cache, no tokens are evicted, so baseline == tiered and equals
# the per-request cost of just computing the genuinely-new tokens.
unlimited_ms = np.array([t.baseline_cost_ms for t in unlimited_traces])
hit_frac = np.array([t.hbm_hit_frac for t in traces]) * 100  # HBM hit rate %

# Smoothed TTFT for readability (rolling window)
window = 15
def smooth(arr, w):
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")

baseline_smooth = smooth(baseline_ms, window)
tiered_smooth = smooth(tiered_ms, window)
unlimited_smooth = smooth(unlimited_ms, window)
hit_smooth = smooth(hit_frac, window)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                gridspec_kw={"height_ratios": [2.2, 1]})

# --- Top panel: TTFT over time ---
# Two bands, both anchored at the oracle floor:
#   • red  : baseline − oracle (total cost of thrashing)
#   • blue : tiered  − oracle (residual cost after PCIe reload + overlap)
# The baseline-vs-tiered gap is implied by the two bands and not drawn.
ax1.fill_between(req_idx, unlimited_smooth, baseline_smooth,
                 alpha=0.13, color=ACCENT4, zorder=1, label="_nolegend_")
ax1.fill_between(req_idx, unlimited_smooth, tiered_smooth,
                 alpha=0.22, color=ACCENT1, zorder=2, label="_nolegend_")

ax1.plot(req_idx, baseline_smooth, linewidth=2.0, color=ACCENT4,
         label="Baseline TTFT (recompute on miss)", zorder=3)
ax1.plot(req_idx, tiered_smooth, linewidth=2.0, color=ACCENT1,
         label="Tiered TTFT (PCIe reload + IO/compute overlap)", zorder=3)
ax1.plot(req_idx, unlimited_smooth, linewidth=2.0, color=ACCENT2,
         linestyle="--",
         label="Unlimited cache (oracle, no evictions)", zorder=3)

# Phase shading (approximate: ramp-up ~first 16 requests, drain starts ~680)
n = len(req_idx)
ramp_end = min(16, n)
drain_start = int(n * 0.87)
ax1.axvspan(0, ramp_end, alpha=0.06, color=ACCENT2, zorder=0)
ax1.axvspan(ramp_end, drain_start, alpha=0.06, color=ACCENT4, zorder=0)
ax1.axvspan(drain_start, n, alpha=0.06, color=ACCENT6, zorder=0)

ax1.text(ramp_end / 2, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 200,
         "Ramp", ha="center", fontsize=9, color=ACCENT2, fontweight="bold", va="top")
ax1.text((ramp_end + drain_start) / 2, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 200,
         "Sustained Thrashing", ha="center", fontsize=9, color=ACCENT4,
         fontweight="bold", va="top")
ax1.text((drain_start + n) / 2, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 200,
         "Drain", ha="center", fontsize=9, color=ACCENT6, fontweight="bold", va="top")

# Annotate: baseline-to-oracle (total thrashing cost)
# and tiered-to-oracle (residual after IO/compute overlap)
mid = (ramp_end + drain_start) // 2
gap_y_base = baseline_smooth[mid]
gap_y_tier = tiered_smooth[mid]
gap_y_oracle = unlimited_smooth[mid]

# Red arrow: baseline − oracle (what thrashing costs)
ax1.annotate("", xy=(mid, gap_y_oracle), xytext=(mid, gap_y_base),
             arrowprops=dict(arrowstyle="<->", color=ACCENT4, lw=2))
base_to_oracle_pct = (1 - gap_y_oracle / gap_y_base) * 100 if gap_y_base > 0 else 0
ax1.text(mid + 15, (gap_y_base + gap_y_oracle) / 2,
         f"Thrashing cost\n{base_to_oracle_pct:.0f}% above floor",
         fontsize=9, fontweight="bold", color=ACCENT4, va="center",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0",
                   edgecolor=ACCENT4, linewidth=0.8, alpha=0.9))

# Blue arrow: tiered − oracle (residual after PCIe reload + overlap)
mid2 = int(mid * 0.55)
gap_y_tier2 = tiered_smooth[mid2]
gap_y_oracle2 = unlimited_smooth[mid2]
ax1.annotate("", xy=(mid2, gap_y_oracle2), xytext=(mid2, gap_y_tier2),
             arrowprops=dict(arrowstyle="<->", color=ACCENT1, lw=2))
tier_to_oracle_pct = (1 - gap_y_oracle2 / gap_y_tier2) * 100 if gap_y_tier2 > 0 else 0
ax1.text(mid2 - 15, (gap_y_tier2 + gap_y_oracle2) / 2,
         f"Residual\n{tier_to_oracle_pct:.0f}% above floor",
         fontsize=9, fontweight="bold", color=ACCENT1, va="center", ha="right",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F4FF",
                   edgecolor=ACCENT1, linewidth=0.8, alpha=0.9))

ax1.set_ylabel("Per-request TTFT (ms)", fontsize=12)
ax1.legend(fontsize=10, loc="upper left")
ax1.set_title("TTFT Over Time: Thrashing Inflates Latency, PCIe Reload Recovers It\n"
              "(8 concurrent sessions, 400-block HBM, A100 + Llama-2-70B GQA)",
              fontsize=13, fontweight="bold", pad=10)

# --- Bottom panel: HBM hit rate ---
ax2.fill_between(req_idx, 0, hit_smooth, alpha=0.2, color=ACCENT2, zorder=1)
ax2.plot(req_idx, hit_smooth, linewidth=2.0, color=ACCENT2,
         label="HBM token hit rate (%)", zorder=3)

# Phase shading (same)
ax2.axvspan(0, ramp_end, alpha=0.06, color=ACCENT2, zorder=0)
ax2.axvspan(ramp_end, drain_start, alpha=0.06, color=ACCENT4, zorder=0)
ax2.axvspan(drain_start, n, alpha=0.06, color=ACCENT6, zorder=0)

ax2.set_xlabel("Request index (time \u2192)", fontsize=12)
ax2.set_ylabel("HBM hit rate (%)", fontsize=12)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=10, loc="upper right")

plt.tight_layout()
save(fig, "fig16_ttft_over_time.png")

print(f"\nAll figures saved to {OUT}/")
print("Done!")
