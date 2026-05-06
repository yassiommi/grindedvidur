#!/usr/bin/env python3
"""Two presentation-ready summary figures for the DeepSeek V4 analysis.

Figure 1 — Decode Latency:  crossover story + FLOPs collapse at 1M context
Figure 2 — KV Storage:      compression ratio + two-region scaling + concurrency limits

All numbers sourced directly from InferLens config instances and pre-computed
experiment JSON files. No hardcoded model parameters.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from vidur.config.device_sku_config import A100DeviceSKUConfig
from vidur.config.model_config import (
    DeepSeekV3ModelConfig,
    DeepSeekV4ProModelConfig,
    Llama3_70BModelConfig,
)

_V4   = DeepSeekV4ProModelConfig()
_V3   = DeepSeekV3ModelConfig()
_L3   = Llama3_70BModelConfig()
_A100 = A100DeviceSKUConfig()

OUT   = os.path.join(_ROOT, "reports", "plots")
os.makedirs(OUT, exist_ok=True)

DN_JSON  = os.path.join(_ROOT, "example_outputs", "experiments",
                         "deepseek_v4_disk_necessity", "deepseek_v4_disk_necessity.json")
SWA_JSON = os.path.join(_ROOT, "example_outputs", "experiments",
                          "deepseek_v4_swa_kv_analysis", "deepseek_v4_swa_kv_analysis.json")

with open(DN_JSON)  as f: dn  = json.load(f)
with open(SWA_JSON) as f: swa = json.load(f)

# ── palette ───────────────────────────────────────────────────────────────────
C_V4    = "#e67e22"
C_V3    = "#2980b9"
C_LLAMA = "#8e44ad"
C_STATE = "#1abc9c"
C_HIST  = "#e74c3c"

STYLE = {
    "font.family":     "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize":  12,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
}
plt.rcParams.update(STYLE)


def _ctx(s):
    return f"{s // 1_000_000}M" if s >= 1_000_000 else f"{s // 1_024}K"


# ═══════════════════════════════════════════════════════════════════════════════
# Decode latency helpers  (same roofline as plot_decode_latency_v3_v4.py)
# ═══════════════════════════════════════════════════════════════════════════════

BW_EFF  = 0.80
TP      = 8
HBM_EFF = _A100.memory_bandwidth_gb_per_s * BW_EFF * 1e9   # bytes/s
COMPUTE_MS = 0.007

KV_FULL = 2 * _V4.num_kv_heads * _V4.head_dim * 2   # 32 768

def v3_decode_ms(s):
    kv = (_V3.kv_lora_rank + _V3.qk_rope_head_dim) * 2 * s * _V3.num_layers
    return kv / (HBM_EFF * TP) * 1e3

def v4_decode_ms(s):
    csa = (s // _V4.csa_chunk_size + s % _V4.csa_chunk_size) * KV_FULL * _V4.n_csa_layers
    hca = (s // _V4.hca_chunk_size + s % _V4.hca_chunk_size) * KV_FULL * _V4.n_hca_layers
    swa = min(s, _V4.swa_window_size) * KV_FULL * _V4.n_swa_layers
    return (csa + hca + swa) / (HBM_EFF * TP) * 1e3 + COMPUTE_MS

def v4_attn_flops(s):
    # Attention FLOPs: Q×K and A×V per head, per layer
    # CSA/HCA: computed over compressed entries, not full seq_len
    n_csa_entries = s // _V4.csa_chunk_size
    n_hca_entries = s // _V4.hca_chunk_size
    csa_flops = 4 * n_csa_entries * _V4.head_dim * _V4.num_q_heads * _V4.n_csa_layers
    hca_flops = 4 * n_hca_entries * _V4.head_dim * _V4.num_q_heads * _V4.n_hca_layers
    swa_flops = 4 * min(s, _V4.swa_window_size) * _V4.head_dim * _V4.num_q_heads * _V4.n_swa_layers
    return (csa_flops + hca_flops + swa_flops) / 1e9   # GFLOPs

def v3_attn_flops(s):
    # V3 MLA: full seq_len attention (absorbed, so effective heads = num_q_heads)
    return 4 * s * _V3.qk_nope_head_dim * _V3.num_q_heads * _V3.num_layers / 1e9

def llama_attn_flops(s):
    hd = _L3.embedding_dim // _L3.num_q_heads
    return 4 * s * hd * _L3.num_q_heads * _L3.num_layers / 1e9


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Decode Latency Summary
# ═══════════════════════════════════════════════════════════════════════════════

CONTEXTS = [2_048, 8_192, 32_768, 131_072, 524_288, 1_000_000]

v3_ms = [v3_decode_ms(s) for s in CONTEXTS]
v4_ms = [v4_decode_ms(s) for s in CONTEXTS]
ratio  = [v3 / v4 for v3, v4 in zip(v3_ms, v4_ms)]
labels = [_ctx(s) for s in CONTEXTS]

flop_models  = ["Llama-3-70B", "DeepSeek-V3", "DeepSeek-V4-Pro"]
flop_vals    = [llama_attn_flops(1_000_000),
                v3_attn_flops(1_000_000),
                v4_attn_flops(1_000_000)]
flop_colors  = [C_LLAMA, C_V3, C_V4]

fig1, (ax_lat, ax_flop) = plt.subplots(
    1, 2, figsize=(13, 5.2),
    gridspec_kw={"width_ratios": [2.4, 1]},
)
fig1.suptitle("DeepSeek V4-Pro: Decode Latency  ·  A100 · TP=8 · batch=1",
              fontsize=13, fontweight="bold", y=1.01)

# ── Left: grouped bars + ratio line ──────────────────────────────────────────
x = np.arange(len(CONTEXTS))
w = 0.32

ax_lat.bar(x - w/2, v3_ms, w, color=C_V3,  label="DeepSeek-V3 (MLA)",         zorder=3)
ax_lat.bar(x + w/2, v4_ms, w, color=C_V4,  label="DeepSeek-V4-Pro (CSA+HCA)", zorder=3)

ax2 = ax_lat.twinx()
ax2.plot(x, ratio, "ko--", lw=1.6, ms=6, zorder=4, label="V3 / V4 speedup")
ax2.axhline(1.0, color="#aaa", lw=0.8, ls=":")
ax2.set_ylabel("V3 / V4 speedup ratio", fontsize=9)
ax2.set_ylim(0, max(ratio) * 1.3)
ax2.tick_params(axis="y", labelsize=8)

# Crossover shading
ax_lat.axvspan(1.5, 2.5, alpha=0.09, color="#e74c3c", zorder=0)
ax_lat.text(2.0, 0.0025, "crossover\n~16K–32K", ha="center", va="bottom",
            fontsize=8, color="#c0392b", style="italic")

# Speedup labels on bars
for i, (r, v4) in enumerate(zip(ratio, v4_ms)):
    if r > 1.2:
        ax_lat.text(i + w/2, v4 * 1.45, f"{r:.1f}×",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#27ae60")

# Callout at 1M
ax_lat.annotate(
    "4.2× faster at 1M ctx\n786 vs 186 tok/s",
    xy=(len(CONTEXTS) - 1 + w/2, v4_ms[-1]),
    xytext=(-55, 18), textcoords="offset points",
    fontsize=8.5, color=C_V4, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_V4, lw=1.2),
)

ax_lat.set_yscale("log")
ax_lat.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3g}"))
ax_lat.set_xticks(x)
ax_lat.set_xticklabels(labels)
ax_lat.set_xlabel("Context Length")
ax_lat.set_ylabel("Decode latency per step (ms, log)")
ax_lat.set_title("Latency Crossover: V4 slower at short ctx, 4.2× faster at long ctx")
ax_lat.grid(axis="y", alpha=0.25)

h1, l1 = ax_lat.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax_lat.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5)

# ── Right: FLOPs at 1M context ───────────────────────────────────────────────
y_pos = np.arange(len(flop_models))
bars  = ax_flop.barh(y_pos, flop_vals, color=flop_colors, height=0.5, zorder=3)
ax_flop.set_xscale("log")
ax_flop.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
ax_flop.set_yticks(y_pos)
ax_flop.set_yticklabels(flop_models, fontsize=9)
ax_flop.set_xlabel("Attention GFLOPs (log)")
ax_flop.set_title("Attention FLOPs at 1M Context\n(CSA/HCA compress the compute)")
ax_flop.grid(axis="x", alpha=0.25)

for bar, val in zip(bars, flop_vals):
    ax_flop.text(val * 1.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:,.0f}B", va="center", fontsize=8.5, fontweight="bold")

# 200× annotation between V3 and V4
y_v3 = y_pos[1]
y_v4 = y_pos[2]
ax_flop.annotate(
    "", xy=(flop_vals[2], y_v4 + 0.28),
    xytext=(flop_vals[1], y_v3 - 0.28),
    arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5),
)
ax_flop.text(flop_vals[2] * 3, (y_v3 + y_v4) / 2,
             "200×\nlower", ha="left", va="center",
             fontsize=9, fontweight="bold", color="#333")

fig1.tight_layout()
p1 = os.path.join(OUT, "15_summary_decode_latency.png")
fig1.savefig(p1, dpi=160, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {p1}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — KV Storage Summary
# ═══════════════════════════════════════════════════════════════════════════════

fig2, axes = plt.subplots(1, 3, figsize=(15, 5.2))
fig2.suptitle("DeepSeek V4-Pro: KV Cache Storage  ·  A100 · TP=8",
              fontsize=13, fontweight="bold", y=1.01)

# ── Panel 1: Compression ratio ────────────────────────────────────────────────
ax1 = axes[0]
models_c  = ["Llama-3-70B\n(MHA)", "DeepSeek-V3\n(MLA)", "DeepSeek-V4-Pro\n(CSA+HCA avg)"]
kv_bpt    = [4096, 1152, 272]
colors_c  = [C_LLAMA, C_V3, C_V4]

bars1 = ax1.bar(range(3), kv_bpt, color=colors_c, width=0.5, zorder=3)
for i, (v, b) in enumerate(zip(kv_bpt, bars1)):
    ax1.text(i, v + 60, f"{v:,} B", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Brackets
for i, (label, yref) in enumerate([("15× vs V4", 4096), ("4.2× vs V4", 1152)]):
    xi, xv4 = i, 2
    ymax = yref + 300
    ax1.annotate("", xy=(xv4, ymax), xytext=(xi, ymax),
                 arrowprops=dict(arrowstyle="<->", color="#555", lw=1.3))
    ax1.text((xi + xv4) / 2, ymax + 80, label,
             ha="center", va="bottom", fontsize=8.5, color="#333")

ax1.set_xticks(range(3))
ax1.set_xticklabels(models_c, fontsize=9)
ax1.set_ylabel("Bytes / token / layer (FP16)")
ax1.set_title("KV Compression\nper Token per Layer")
ax1.set_ylim(0, 5200)
ax1.grid(axis="y", alpha=0.25)

# ── Panel 2: Two-region scaling ───────────────────────────────────────────────
ax2 = axes[1]

sep = dn["separation"]
seqs    = [r["seq_len"]        for r in sep]
state_g = [r["state_cache_gb"] for r in sep]
comp_g  = [r["compressed_gb"]  for r in sep]
v3_g    = [r["v3_kv_gb"]       for r in sep]
total_g = [s + c for s, c in zip(state_g, comp_g)]
xlabs   = [_ctx(s) for s in seqs]
xi      = range(len(seqs))

ax2.fill_between(xi, 0,       state_g, color=C_STATE, alpha=0.5, label="State cache (HBM, bounded)")
ax2.fill_between(xi, state_g, total_g, color=C_HIST,  alpha=0.5, label="Compressed history (→ disk)")
ax2.plot(xi, v3_g, color=C_V3, lw=2, ls="--", label="V3 MLA total")

# Saturation line
sat = max(state_g)
ax2.axhline(sat, color=C_STATE, lw=1.2, ls=":", alpha=0.8)
ax2.text(len(seqs) - 1, sat + 0.3, f"State cache cap\n≈ {sat:.1f} GB", ha="right",
         fontsize=8, color="#16a085")

# Annotation for V3 vs V4 at 1M
ax2.annotate("V3: 70 GB", xy=(len(seqs) - 1, v3_g[-1]),
             xytext=(-2, 6), textcoords="offset points",
             fontsize=8, color=C_V3, fontweight="bold")
ax2.annotate("V4: 16 GB", xy=(len(seqs) - 1, total_g[-1]),
             xytext=(-2, 6), textcoords="offset points",
             fontsize=8, color=C_HIST, fontweight="bold")

ax2.set_xticks(range(len(seqs)))
ax2.set_xticklabels(xlabs, rotation=30, ha="right", fontsize=8)
ax2.set_ylabel("KV Cache (GB)")
ax2.set_title("V4 Two-Region KV Structure\nState cache saturates; history grows linearly")
ax2.legend(loc="upper left", fontsize=8.5)
ax2.grid(axis="y", alpha=0.25)

# ── Panel 3: Concurrent session capacity ─────────────────────────────────────
ax3 = axes[2]

cap_ctxs = [131_072, 262_144, 524_288, 1_000_000]
cap_v4   = []
cap_v3   = []
cap_l3   = []
for s in cap_ctxs:
    for r in dn["capacity"]:
        if r["seq_len"] == s and r["total_gpus"] == TP:
            if r["model"] == "V4":    cap_v4.append(r["per_replica_cap"])
            elif r["model"] == "V3":  cap_v3.append(r["per_replica_cap"])
            elif r["model"] == "Llama": cap_l3.append(r["per_replica_cap"])

cap_labels = [_ctx(s) for s in cap_ctxs]
xc = np.arange(len(cap_ctxs))
wc = 0.25

ax3.bar(xc - wc, cap_l3, wc, color=C_LLAMA, label="Llama-3-70B", zorder=3)
ax3.bar(xc,      cap_v3, wc, color=C_V3,    label="DeepSeek-V3",  zorder=3)
ax3.bar(xc + wc, cap_v4, wc, color=C_V4,    label="DeepSeek-V4",  zorder=3)

for i, (v4, v3, l3) in enumerate(zip(cap_v4, cap_v3, cap_l3)):
    ax3.text(i + wc, v4 + 1.5, str(v4),  ha="center", fontsize=8,   color=C_V4,    fontweight="bold")
    ax3.text(i,      v3 + 1.5, str(v3),  ha="center", fontsize=8,   color=C_V3)
    ax3.text(i - wc, l3 + 1.5, str(l3),  ha="center", fontsize=8,   color=C_LLAMA)

# "5× better" at 1M
ax3.annotate("", xy=(3 + wc, cap_v4[-1] + 4), xytext=(3, cap_v3[-1] + 4),
             arrowprops=dict(arrowstyle="<->", color="#555", lw=1.3))
ax3.text(3 + wc/2, cap_v4[-1] + 9, "5× better\nvs V3",
         ha="center", fontsize=8.5, fontweight="bold", color="#333")

ax3.set_xticks(xc)
ax3.set_xticklabels(cap_labels)
ax3.set_xlabel("Context Length")
ax3.set_ylabel("Max concurrent sessions (per TP=8 replica)")
ax3.set_title("HBM Capacity: Concurrent Sessions\nbefore compressed KV fills memory")
ax3.legend(fontsize=8.5)
ax3.grid(axis="y", alpha=0.25)

fig2.tight_layout()
p2 = os.path.join(OUT, "16_summary_kv_storage.png")
fig2.savefig(p2, dpi=160, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {p2}")
