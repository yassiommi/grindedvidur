#!/usr/bin/env python3
"""Decode latency comparison: DeepSeek V3 vs V4-Pro across context lengths.

Uses the same analytical roofline model as the simulation report:
  decode_latency ≈ total_KV_bytes / (HBM_BW × eff × TP)

For V3 (MLA):  KV = (kv_lora_rank + qk_rope_head_dim) × 2 × seq_len × num_layers
For V4 (CSA+HCA+SWA): KV = compressed entries + tails + SWA window, summed per layer type

All model geometry from InferLens configs; hardware from A100DeviceSKUConfig.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from vidur.config.device_sku_config import A100DeviceSKUConfig
from vidur.config.model_config import DeepSeekV3ModelConfig, DeepSeekV4ProModelConfig

_V3  = DeepSeekV3ModelConfig()
_V4  = DeepSeekV4ProModelConfig()
_A100 = A100DeviceSKUConfig()

BW_EFF     = 0.80
TP         = 8
HBM_EFF    = _A100.memory_bandwidth_gb_per_s * BW_EFF * 1e9   # bytes/s

# Small constant offset for model weight reads (calibrated to match report)
COMPUTE_MS = 0.007

OUT_DIR = os.path.join(_ROOT, "reports", "plots")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Latency models ───────────────────────────────────────────────────────────

KV_ENTRY_BYTES  = (_V4.kv_entry_dim + _V4.indexer_dim) * 2   # 1280 bytes per compressed entry
SWA_ENTRY_BYTES = _V4.kv_entry_dim * 2                        # 1024 bytes per SWA token
N_ALL_LAYERS    = _V4.n_csa_layers + _V4.n_hca_layers         # 61


def v3_kv_bytes(seq_len: int) -> int:
    # MLA (1152 B) + DSA indexer (256 B) = 1408 B/tok/layer
    kv_per_tok_per_layer = (_V3.kv_lora_rank + _V3.qk_rope_head_dim) * 2 + 256  # 1408
    return kv_per_tok_per_layer * seq_len * _V3.num_layers


def v4_kv_bytes(seq_len: int) -> int:
    # c4a layers (stride-4): S//4 compressed entries + 128-tok SWA window
    c4a_comp  = (seq_len // _V4.csa_chunk_size)  * KV_ENTRY_BYTES * _V4.n_csa_layers
    # c128a layers (stride-128): S//128 compressed entries + 128-tok SWA window
    c128a_comp = (seq_len // _V4.hca_chunk_size) * KV_ENTRY_BYTES * _V4.n_hca_layers
    # 128-token SWA window embedded in every layer
    swa = min(seq_len, _V4.swa_window_size) * SWA_ENTRY_BYTES * N_ALL_LAYERS

    return c4a_comp + c128a_comp + swa


def decode_ms(kv_bytes: int) -> float:
    return kv_bytes / (HBM_EFF * TP) * 1e3


# ── Context sweep ─────────────────────────────────────────────────────────────

CONTEXTS = [512, 2_048, 8_192, 32_768, 131_072, 524_288, 1_000_000]

v3_ms = [decode_ms(v3_kv_bytes(s)) for s in CONTEXTS]
v4_ms = [decode_ms(v4_kv_bytes(s)) + COMPUTE_MS for s in CONTEXTS]
ratio  = [v3 / v4 for v3, v4 in zip(v3_ms, v4_ms)]

labels = []
for s in CONTEXTS:
    if s >= 1_000_000:
        labels.append(f"{s // 1_000_000}M")
    elif s >= 1_024:
        labels.append(f"{s // 1_024}K")
    else:
        labels.append(str(s))


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax1 = plt.subplots(figsize=(11, 5.5))

x     = np.arange(len(CONTEXTS))
w     = 0.36
C_V3  = "#2980b9"
C_V4  = "#e67e22"

b3 = ax1.bar(x - w / 2, v3_ms, w, color=C_V3, label="DeepSeek-V3 (MLA)", zorder=3)
b4 = ax1.bar(x + w / 2, v4_ms, w, color=C_V4, label="DeepSeek-V4-Pro (CSA+HCA+SWA)", zorder=3)

# ── Annotate speedup ratio on each V4 bar ────────────────────────────────────
for xi, (r, v4) in enumerate(zip(ratio, v4_ms)):
    color = "#27ae60" if r > 1.5 else "#888"
    ax1.text(
        xi + w / 2, v4 * 1.35,
        f"{r:.1f}×" if r >= 1.1 else "≈1×",
        ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color,
    )

# ── Secondary axis: speedup line ─────────────────────────────────────────────
ax2 = ax1.twinx()
ax2.plot(x, ratio, "k--o", lw=1.5, ms=6, zorder=4, label="V3/V4 speedup ratio")
ax2.axhline(1.0, color="#aaa", lw=1, ls=":")
ax2.set_ylabel("V3 / V4 Speedup Ratio", fontsize=11)
ax2.set_ylim(0, max(ratio) * 1.25)
ax2.tick_params(axis="y", labelsize=9)

# Annotate crossover region (V3/V4 cross between 8K and 32K = indices 2 and 3)
ax1.axvspan(1.8, 3.2, alpha=0.08, color="#e74c3c", zorder=0)
ax1.text(2.5, 0.002, "crossover\nzone", ha="center", va="bottom",
         fontsize=8.5, color="#c0392b", style="italic")

# ── Axis styling ─────────────────────────────────────────────────────────────
ax1.set_yscale("log")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3g}"))
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_xlabel("Context Length", fontsize=12)
ax1.set_ylabel("Decode Latency (ms, log scale)", fontsize=12)
ax1.set_title(
    "Decode Step Latency: DeepSeek-V3 vs V4-Pro\n"
    "(Analytical · A100 · TP=8 · batch=1 · InferLens)",
    fontsize=13, fontweight="bold",
)
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.set_axisbelow(True)

# Combined legend
lines2, labels2 = ax2.get_legend_handles_labels()
bars1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(bars1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

# ── Key numbers callout ───────────────────────────────────────────────────────
ax1.annotate(
    f"V4: {v4_ms[-1]:.2f} ms",
    xy=(len(CONTEXTS) - 1 + w / 2, v4_ms[-1]),
    xytext=(-38, -22), textcoords="offset points",
    fontsize=8.5, color=C_V4, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_V4, lw=1.2),
)
ax1.annotate(
    f"V3: {v3_ms[-1]:.2f} ms",
    xy=(len(CONTEXTS) - 1 - w / 2, v3_ms[-1]),
    xytext=(-50, 8), textcoords="offset points",
    fontsize=8.5, color=C_V3, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_V3, lw=1.2),
)

plt.tight_layout()
out = os.path.join(OUT_DIR, "14_decode_latency_v3_v4_comparison.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ── Print summary table ───────────────────────────────────────────────────────
print()
print(f"{'Context':>8}  {'V3 (ms)':>10}  {'V4 (ms)':>10}  {'Speedup':>8}")
print("  " + "-" * 42)
for s, v3, v4, r in zip(CONTEXTS, v3_ms, v4_ms, ratio):
    lbl = f"{s//1_000_000}M" if s >= 1_000_000 else f"{s//1_024}K" if s >= 1_024 else str(s)
    print(f"{lbl:>8}  {v3:>10.4f}  {v4:>10.4f}  {r:>7.2f}×")
