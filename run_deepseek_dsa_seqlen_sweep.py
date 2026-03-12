#!/usr/bin/env python3
"""DeepSeek Sparse Attention (DSA) vs MLA: Sequence Length Sweep.

Compares KV cache size and decode-step timing between:
  - MLA (full attention): loads the entire compressed KV cache for all seq_len tokens
  - DSA (DeepSeek Sparse Attention): uses a lightning indexer to select top-k tokens,
    then runs full MLA only on the selected subset + sliding window

DSA architecture (from DeepSeek-V3.2):
  - Lightning indexer: dedicated FP8 K cache, scores all previous tokens cheaply
  - Token selector: picks top-k=2048 tokens based on indexer scores
  - Sliding window: local context window of 512 tokens (always attended)
  - Full MLA attention runs only on selected + sliding window tokens (~2560)

Hardware assumptions:
  - A100 GPU, PCIe Gen4 (31.5 GB/s), HBM2e (2039 GB/s)
  - DGX A100 (NVLink 600 GB/s intra-node)
  - TP=4, EP=4, bandwidth efficiency factor: 0.8
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Output directory ──────────────────────────────────────────────────
RESULTS_DIR = "example_outputs/experiments/deepseek_dsa_seqlen_sweep"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hardware specs (A100, PCIe Gen4) ─────────────────────────────────
BW_EFFICIENCY = 0.8
PCIE_BW_GBS = 31.5
HBM_BW_GBS = 2039.0
NVLINK_BW_GBS = 600.0

PCIE_EFF_BPS = PCIE_BW_GBS * BW_EFFICIENCY * (1024 ** 3)
HBM_EFF_BPS = HBM_BW_GBS * BW_EFFICIENCY * (1024 ** 3)

# ── DeepSeek-V3 MLA model config ────────────────────────────────────
NUM_LAYERS = 61
HIDDEN_SIZE = 7168
NUM_Q_HEADS = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
BYTES_PER_PARAM = 2  # FP16

# MoE config
NUM_ROUTED_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8

# Parallelism
TP = 4
EP = 4

# MLA KV cache: compressed latent per token per layer (FP16)
MLA_KV_BYTES_PER_TOKEN = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * BYTES_PER_PARAM  # 1152 bytes

# ── DSA-specific config ──────────────────────────────────────────────
# Lightning indexer: dedicated FP8 K cache for fast scoring.
# The indexer operates in the compressed MLA latent space (kv_lora_rank dims)
# with FP8 precision (1 byte per param).
INDEXER_K_BYTES_PER_TOKEN = KV_LORA_RANK * 1  # 512 bytes (FP8, kv_lora_rank dims)

# Token selection: top-k tokens selected per query
DSA_SELECTED_TOKENS = 2048

# Sliding window: local context always attended
DSA_SLIDING_WINDOW = 512

# Total tokens that get full MLA attention in DSA
DSA_ATTENDED_TOKENS = DSA_SELECTED_TOKENS + DSA_SLIDING_WINDOW  # 2560

# ── Profiling-calibrated constants (from TP=4 runs) ─────────────────
ATTN_PROJ_CONSTANT_MS = 0.38   # per layer (2x heads per GPU → ~2x proj)
MOE_COMPUTE_CONSTANT_MS = 0.95  # per layer (2x experts per GPU → ~2x compute)
TP_COMM_PER_LAYER_MS = 0.22    # fewer participants → less comm overhead
EP_COMM_PER_LAYER_MS = 0.08    # EP=4: more tokens routed per link

# DSA indexer compute overhead per layer (ms):
# The lightning indexer does a cheap FP8 matmul (query × indexer_K^T) plus
# top-k selection. FP8 with small head count makes this very fast.
# Estimated ~0.05ms per layer for the indexer at BS=1.
DSA_INDEXER_COMPUTE_MS = 0.05


# ── Timing models ────────────────────────────────────────────────────
def mla_times_per_layer(seq_len: int) -> dict:
    """MLA (full attention) timing per layer for a single decode step (BS=1)."""
    kv_bytes = MLA_KV_BYTES_PER_TOKEN * seq_len
    kv_load_ms = (kv_bytes / PCIE_EFF_BPS) * 1e3

    attn_core_ms = (kv_bytes / HBM_EFF_BPS) * 1e3
    compute_ms = ATTN_PROJ_CONSTANT_MS + attn_core_ms + MOE_COMPUTE_CONSTANT_MS

    comm_ms = TP_COMM_PER_LAYER_MS + EP_COMM_PER_LAYER_MS

    return {
        "kv_load_ms": kv_load_ms,
        "compute_ms": compute_ms,
        "comm_ms": comm_ms,
        "kv_bytes": kv_bytes,
    }


def dsa_times_per_layer(seq_len: int) -> dict:
    """DSA (sparse attention) timing per layer for a single decode step (BS=1).

    DSA decode step:
      1. Load indexer K cache (FP8, all seq_len tokens) from host → GPU
      2. Run indexer: cheap FP8 matmul to score all tokens, top-k selection
      3. Load full MLA KV only for selected + sliding window tokens
      4. Run full MLA attention on the subset
      5. MoE compute (same as MLA)
      6. Communication (same as MLA)
    """
    # Effective tokens for sliding window (capped at seq_len)
    sliding = min(DSA_SLIDING_WINDOW, seq_len)
    # Selected tokens can't exceed available tokens minus sliding
    selected = min(DSA_SELECTED_TOKENS, max(0, seq_len - sliding))
    attended = selected + sliding

    # 1. KV cache load from host → GPU
    #    a) Indexer K cache: FP8, all seq_len tokens
    indexer_kv_bytes = INDEXER_K_BYTES_PER_TOKEN * seq_len
    #    b) Full MLA KV: only for attended tokens
    mla_kv_bytes = MLA_KV_BYTES_PER_TOKEN * attended
    #    Total KV bytes transferred
    total_kv_bytes = indexer_kv_bytes + mla_kv_bytes
    kv_load_ms = (total_kv_bytes / PCIE_EFF_BPS) * 1e3

    # 2. Compute
    #    a) Indexer: FP8 matmul (query × all indexer K) + top-k
    #       Scales with seq_len but is very cheap (FP8, few heads)
    #       At BS=1, this is bandwidth-bound: reading indexer K cache from HBM
    indexer_hbm_ms = (indexer_kv_bytes / HBM_EFF_BPS) * 1e3
    indexer_ms = DSA_INDEXER_COMPUTE_MS + indexer_hbm_ms

    #    b) Attention projections (same as MLA, constant)
    #    c) Attention core: reads only attended tokens' KV from HBM
    attn_core_ms = (mla_kv_bytes / HBM_EFF_BPS) * 1e3

    #    d) MoE compute (same as MLA)
    compute_ms = indexer_ms + ATTN_PROJ_CONSTANT_MS + attn_core_ms + MOE_COMPUTE_CONSTANT_MS

    # 3. Communication (same as MLA — output activation size unchanged)
    comm_ms = TP_COMM_PER_LAYER_MS + EP_COMM_PER_LAYER_MS

    return {
        "kv_load_ms": kv_load_ms,
        "compute_ms": compute_ms,
        "comm_ms": comm_ms,
        "kv_bytes": total_kv_bytes,
        "indexer_kv_bytes": indexer_kv_bytes,
        "mla_kv_bytes": mla_kv_bytes,
        "attended_tokens": attended,
    }


def compute_total(seq_len: int, method: str) -> dict:
    """Total time across all layers for one decode step."""
    if method == "MLA":
        per_layer = mla_times_per_layer(seq_len)
    else:
        per_layer = dsa_times_per_layer(seq_len)

    result = {
        "seq_len": seq_len,
        "method": method,
        "kv_load_ms": per_layer["kv_load_ms"] * NUM_LAYERS,
        "compute_ms": per_layer["compute_ms"] * NUM_LAYERS,
        "comm_ms": per_layer["comm_ms"] * NUM_LAYERS,
        "kv_bytes_per_layer": per_layer["kv_bytes"],
        "kv_gb_total": per_layer["kv_bytes"] * NUM_LAYERS / (1024 ** 3),
    }
    if method == "DSA":
        result["indexer_kv_bytes_per_layer"] = per_layer["indexer_kv_bytes"]
        result["mla_kv_bytes_per_layer"] = per_layer["mla_kv_bytes"]
        result["attended_tokens"] = per_layer["attended_tokens"]
    return result


# ── Sweep ─────────────────────────────────────────────────────────────
SEQ_LENS = [
    128 * 1024,       # 128K
    256 * 1024,       # 256K
    512 * 1024,       # 512K
    1 * 1024 * 1024,  # 1M
    2 * 1024 * 1024,  # 2M
    4 * 1024 * 1024,  # 4M
    8 * 1024 * 1024,  # 8M
    10 * 1024 * 1024, # 10M
]

mla_results = [compute_total(s, "MLA") for s in SEQ_LENS]
dsa_results = [compute_total(s, "DSA") for s in SEQ_LENS]

# ── Save results ──────────────────────────────────────────────────────
summary = {
    "description": "DeepSeek-V3 DSA vs MLA: KV cache size and decode timing comparison",
    "model": "deepseek-ai/DeepSeek-V3.2",
    "device": "A100 (PCIe Gen4, 31.5 GB/s)",
    "parallelism": f"TP={TP}, EP={EP}",
    "batch_size": 1,
    "mla_config": {
        "kv_bytes_per_token_per_layer": MLA_KV_BYTES_PER_TOKEN,
        "attention": "full (all seq_len tokens)",
    },
    "dsa_config": {
        "indexer_k_bytes_per_token_per_layer": INDEXER_K_BYTES_PER_TOKEN,
        "mla_kv_bytes_per_token_per_layer": MLA_KV_BYTES_PER_TOKEN,
        "selected_tokens": DSA_SELECTED_TOKENS,
        "sliding_window": DSA_SLIDING_WINDOW,
        "total_attended": DSA_ATTENDED_TOKENS,
    },
    "num_layers": NUM_LAYERS,
    "mla_results": mla_results,
    "dsa_results": dsa_results,
}

with open(os.path.join(RESULTS_DIR, "dsa_vs_mla_results.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

# ── Print comparison tables ──────────────────────────────────────────
def seq_label(s):
    return f"{s // 1024}K" if s < 1024 * 1024 else f"{s // (1024 * 1024)}M"


print()
print("=" * 110)
print("DeepSeek-V3: DSA vs MLA — KV Cache Size Comparison (per layer)")
print("=" * 110)
print(f"{'Seq Len':>10s}  {'MLA KV (MB)':>12s}  {'DSA KV (MB)':>12s}  "
      f"{'DSA Idx (MB)':>13s}  {'DSA MLA (MB)':>13s}  {'Reduction':>10s}  {'Attended':>10s}")
print("-" * 110)

for m, d in zip(mla_results, dsa_results):
    mla_mb = m["kv_bytes_per_layer"] / (1024 ** 2)
    dsa_mb = d["kv_bytes_per_layer"] / (1024 ** 2)
    idx_mb = d["indexer_kv_bytes_per_layer"] / (1024 ** 2)
    mla_sub_mb = d["mla_kv_bytes_per_layer"] / (1024 ** 2)
    reduction = mla_mb / dsa_mb
    print(f"{seq_label(m['seq_len']):>10s}  {mla_mb:>12.1f}  {dsa_mb:>12.1f}  "
          f"{idx_mb:>13.1f}  {mla_sub_mb:>13.1f}  {reduction:>9.1f}x  "
          f"{d['attended_tokens']:>10,d}")

print("-" * 110)
print(f"\nMLA KV: {MLA_KV_BYTES_PER_TOKEN} B/token/layer (FP16, kv_lora_rank={KV_LORA_RANK} + rope={QK_ROPE_HEAD_DIM})")
print(f"DSA indexer K: {INDEXER_K_BYTES_PER_TOKEN} B/token/layer (FP8, kv_lora_rank={KV_LORA_RANK})")
print(f"DSA selected: {DSA_SELECTED_TOKENS} tokens | sliding window: {DSA_SLIDING_WINDOW} tokens")

print()
print("=" * 110)
print(f"DeepSeek-V3: DSA vs MLA — Decode Step Timing Comparison (BS=1, TP={TP}, EP={EP})")
print("=" * 110)
print(f"{'Seq Len':>10s}  {'MLA Total':>12s}  {'DSA Total':>12s}  "
      f"{'Speedup':>8s}  {'MLA KV Load':>12s}  {'DSA KV Load':>12s}  "
      f"{'MLA Compute':>12s}  {'DSA Compute':>12s}")
print("-" * 110)

for m, d in zip(mla_results, dsa_results):
    m_total = m["kv_load_ms"] + m["compute_ms"] + m["comm_ms"]
    d_total = d["kv_load_ms"] + d["compute_ms"] + d["comm_ms"]
    speedup = m_total / d_total

    def fmt(ms):
        return f"{ms / 1000:.2f}s" if ms >= 1000 else f"{ms:.1f}ms"

    print(f"{seq_label(m['seq_len']):>10s}  {fmt(m_total):>12s}  {fmt(d_total):>12s}  "
          f"{speedup:>7.1f}x  {fmt(m['kv_load_ms']):>12s}  {fmt(d['kv_load_ms']):>12s}  "
          f"{fmt(m['compute_ms']):>12s}  {fmt(d['compute_ms']):>12s}")

print("-" * 110)

# ── Plot 1: KV cache size comparison ─────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})

C_MLA = "#3498db"
C_DSA_IDX = "#e74c3c"
C_DSA_MLA = "#f39c12"
C_DSA_TOTAL = "#2ecc71"

seq_labels = [seq_label(s) for s in SEQ_LENS]
x = np.arange(len(SEQ_LENS))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: KV cache size per layer
mla_kv_mb = [m["kv_bytes_per_layer"] / (1024 ** 2) for m in mla_results]
dsa_kv_mb = [d["kv_bytes_per_layer"] / (1024 ** 2) for d in dsa_results]
dsa_idx_mb = [d["indexer_kv_bytes_per_layer"] / (1024 ** 2) for d in dsa_results]
dsa_mla_mb = [d["mla_kv_bytes_per_layer"] / (1024 ** 2) for d in dsa_results]

bar_w = 0.35
bars1 = ax1.bar(x - bar_w / 2, mla_kv_mb, bar_w, label="MLA (full)", color=C_MLA,
                edgecolor="black", linewidth=0.5, alpha=0.85)
# Stacked bar for DSA: indexer + MLA subset
bars2a = ax1.bar(x + bar_w / 2, dsa_idx_mb, bar_w, label="DSA: Indexer K (FP8)",
                 color=C_DSA_IDX, edgecolor="black", linewidth=0.5, alpha=0.85)
bars2b = ax1.bar(x + bar_w / 2, dsa_mla_mb, bar_w, bottom=dsa_idx_mb,
                 label=f"DSA: MLA KV ({DSA_ATTENDED_TOKENS} tokens)",
                 color=C_DSA_MLA, edgecolor="black", linewidth=0.5, alpha=0.85)

ax1.set_yscale("log")
ax1.set_xlabel("Sequence Length")
ax1.set_ylabel("KV Cache per Layer (MB, log scale)")
ax1.set_title("KV Cache Size: MLA vs DSA", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(seq_labels)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(axis="y", alpha=0.3, which="both")

# Add reduction labels
for i in range(len(SEQ_LENS)):
    reduction = mla_kv_mb[i] / dsa_kv_mb[i]
    ax1.text(x[i] + bar_w / 2, dsa_kv_mb[i] * 1.3,
             f"{reduction:.1f}x", ha="center", fontsize=8, fontweight="bold",
             color="#27ae60")

# Right: Decode step timing comparison
mla_totals = [m["kv_load_ms"] + m["compute_ms"] + m["comm_ms"] for m in mla_results]
dsa_totals = [d["kv_load_ms"] + d["compute_ms"] + d["comm_ms"] for d in dsa_results]

# Stacked bars: KV load + compute + comm
bar_w2 = 0.35
# MLA stacked
ax2.bar(x - bar_w2 / 2, [m["kv_load_ms"] for m in mla_results], bar_w2,
        label="MLA: KV Load", color="#3498db", edgecolor="black", linewidth=0.5, alpha=0.85)
ax2.bar(x - bar_w2 / 2, [m["compute_ms"] for m in mla_results], bar_w2,
        bottom=[m["kv_load_ms"] for m in mla_results],
        label="MLA: Compute", color="#85c1e9", edgecolor="black", linewidth=0.5, alpha=0.85)

# DSA stacked
ax2.bar(x + bar_w2 / 2, [d["kv_load_ms"] for d in dsa_results], bar_w2,
        label="DSA: KV Load", color="#2ecc71", edgecolor="black", linewidth=0.5, alpha=0.85)
ax2.bar(x + bar_w2 / 2, [d["compute_ms"] for d in dsa_results], bar_w2,
        bottom=[d["kv_load_ms"] for d in dsa_results],
        label="DSA: Compute", color="#82e0aa", edgecolor="black", linewidth=0.5, alpha=0.85)

ax2.set_yscale("log")
ax2.set_xlabel("Sequence Length")
ax2.set_ylabel("Total Decode Step Time (ms, log scale)")
ax2.set_title("Decode Timing: MLA vs DSA", fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(seq_labels)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(axis="y", alpha=0.3, which="both")

# Add speedup labels
for i in range(len(SEQ_LENS)):
    speedup = mla_totals[i] / dsa_totals[i]
    ax2.text(x[i], max(mla_totals[i], dsa_totals[i]) * 1.3,
             f"{speedup:.1f}x", ha="center", fontsize=8, fontweight="bold",
             color="#27ae60")

fig.suptitle(
    "DeepSeek-V3: MLA vs DSA (Sparse Attention)\n"
    f"BS=1, A100 PCIe Gen4, TP={TP}, EP={EP} | "
    f"DSA: top-{DSA_SELECTED_TOKENS} selected + {DSA_SLIDING_WINDOW} sliding window",
    fontweight="bold", fontsize=13, y=1.03,
)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "dsa_vs_mla_comparison.png")
plt.savefig(plot_path, bbox_inches="tight")
plt.close()

# ── Plot 2: DSA KV cache breakdown (stacked area) ───────────────────
fig2, ax3 = plt.subplots(figsize=(10, 6))

# Show how DSA KV cache is composed
indexer_pct = [d["indexer_kv_bytes_per_layer"] / d["kv_bytes_per_layer"] * 100
               for d in dsa_results]
mla_sub_pct = [d["mla_kv_bytes_per_layer"] / d["kv_bytes_per_layer"] * 100
               for d in dsa_results]

ax3.bar(x, indexer_pct, 0.6, label="Indexer K (FP8, scales with seq_len)",
        color=C_DSA_IDX, edgecolor="black", linewidth=0.5, alpha=0.85)
ax3.bar(x, mla_sub_pct, 0.6, bottom=indexer_pct,
        label=f"MLA KV (FP16, fixed {DSA_ATTENDED_TOKENS} tokens)",
        color=C_DSA_MLA, edgecolor="black", linewidth=0.5, alpha=0.85)

ax3.set_xlabel("Sequence Length")
ax3.set_ylabel("% of DSA KV Cache")
ax3.set_title("DSA KV Cache Composition: Indexer vs MLA Subset", fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(seq_labels)
ax3.legend(loc="center right", fontsize=10)
ax3.set_ylim(0, 105)
ax3.grid(axis="y", alpha=0.3)

# Annotate with absolute sizes
for i in range(len(SEQ_LENS)):
    total_mb = dsa_kv_mb[i]
    ax3.text(x[i], 102, f"{total_mb:.0f} MB", ha="center", fontsize=8,
             fontweight="bold", color="#2c3e50")

plt.tight_layout()
plot2_path = os.path.join(RESULTS_DIR, "dsa_kv_breakdown.png")
plt.savefig(plot2_path, bbox_inches="tight")
plt.close()

print(f"\nPlots saved to {RESULTS_DIR}/")
print(f"  - {plot_path}")
print(f"  - {plot2_path}")
