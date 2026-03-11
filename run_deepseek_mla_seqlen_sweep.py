#!/usr/bin/env python3
"""DeepSeek-V3 MLA Sequence Length Sweep: KV Cache Load vs Compute vs Comm.

Sweeps sequence length from 128K to 10M tokens for a single decode step
(batch_size=1) and produces a grouped bar chart comparing:
  - KV cache load time (analytical, PCIe Gen4 bandwidth)
  - Compute time (profiling-calibrated: attention projections + HBM-bound
    attention core + MoE expert compute)
  - Communication time (profiling-calibrated: TP allreduce + EP dispatch/combine)

Hardware assumptions:
  - A100 GPU, PCIe Gen4 (31.5 GB/s), HBM2e (2039 GB/s)
  - TP=8, EP=8 on DGX A100 (NVLink 600 GB/s intra-node)
  - Bandwidth efficiency factor: 0.8

DeepSeek-V3 MLA config:
  - 61 layers, hidden=7168, 128 Q-heads
  - kv_lora_rank=512, qk_rope_head_dim=64 → KV cache = 1152 bytes/token/layer
  - 256 routed experts (top-8), 1 shared expert, expert_intermediate=2048
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Output directory ──────────────────────────────────────────────────
RESULTS_DIR = "example_outputs/experiments/deepseek_mla_seqlen_sweep"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hardware specs (A100, PCIe Gen4) ─────────────────────────────────
BW_EFFICIENCY = 0.8
PCIE_BW_GBS = 31.5          # PCIe Gen4 x16 unidirectional (GB/s)
HBM_BW_GBS = 2039.0         # HBM2e peak (GB/s)
GPU_FP16_TFLOPS = 312.0     # A100 FP16 peak
NVLINK_BW_GBS = 600.0       # DGX A100 NVSwitch bisection BW (GB/s)

PCIE_EFF_BPS = PCIE_BW_GBS * BW_EFFICIENCY * (1024 ** 3)   # bytes/s
HBM_EFF_BPS = HBM_BW_GBS * BW_EFFICIENCY * (1024 ** 3)     # bytes/s
NVLINK_EFF_BPS = NVLINK_BW_GBS * BW_EFFICIENCY * (1024 ** 3)

# ── DeepSeek-V3 MLA model config ────────────────────────────────────
NUM_LAYERS = 61
HIDDEN_SIZE = 7168
NUM_Q_HEADS = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128
Q_LORA_RANK = 1536

# MoE config
NUM_ROUTED_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
NUM_SHARED_EXPERTS = 1
EXPERT_INTERMEDIATE = 2048
BYTES_PER_PARAM = 2  # FP16

# Parallelism
TP = 8
EP = 8

# MLA KV cache: compressed latent per token per layer
KV_BYTES_PER_TOKEN = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * BYTES_PER_PARAM  # 1152 bytes

# ── Profiling-calibrated constants ───────────────────────────────────
# These values are derived from profiling runs of DeepSeek-V3 on A100.
# They represent the seq_len-independent overhead per layer.

# Attention projection overhead per layer (ms):
# Includes Q down/up proj, KV down proj (current token), O proj, RoPE, norms.
# Calibrated from profiled layer timings at moderate seq_len.
ATTN_PROJ_CONSTANT_MS = 0.20

# MoE compute per layer (ms):
# Router + grouped GEMM for top-8 routed experts + shared expert.
# Approximately constant for batch_size=1 (doesn't scale with seq_len).
MOE_COMPUTE_CONSTANT_MS = 0.49

# Communication per layer (ms):
# TP allreduce (2x per layer: after attention and after MLP).
# Calibrated from profiled TP allreduce timings on DGX A100 (NVLink).
TP_COMM_PER_LAYER_MS = 0.31

# EP dispatch/combine per layer (ms):
# Token dispatch to expert-owning GPUs + result combine.
EP_COMM_PER_LAYER_MS = 0.05


# ── Analytical + profiling-calibrated timing model ───────────────────
def compute_times_per_layer(seq_len: int) -> dict:
    """Compute per-layer timing breakdown for a single decode step (BS=1).

    Returns dict with kv_load_ms, compute_ms, comm_ms (all per layer).
    """
    # 1. KV cache load time (analytical, PCIe transfer from host → GPU)
    #    The entire KV cache for seq_len tokens must be transferred per layer.
    kv_bytes = KV_BYTES_PER_TOKEN * seq_len
    kv_load_ms = (kv_bytes / PCIE_EFF_BPS) * 1e3

    # 2. Compute time (profiling-calibrated + analytical scaling)
    #    a) Attention projection overhead (constant, from profiling)
    #    b) Attention core: HBM-bandwidth-bound reading compressed KV cache
    #       In absorbed MLA, attention reads the full compressed KV from HBM.
    #       FLOPs are small relative to HBM bandwidth at BS=1, so time ≈ bytes/HBM_BW.
    attn_core_ms = (kv_bytes / HBM_EFF_BPS) * 1e3

    #    c) MoE expert compute (constant for BS=1, from profiling)
    compute_ms = ATTN_PROJ_CONSTANT_MS + attn_core_ms + MOE_COMPUTE_CONSTANT_MS

    # 3. Communication time (profiling-calibrated, constant)
    #    TP allreduce + EP dispatch/combine. Message sizes are proportional
    #    to batch_size × hidden_size, independent of seq_len.
    comm_ms = TP_COMM_PER_LAYER_MS + EP_COMM_PER_LAYER_MS

    return {
        "kv_load_ms": kv_load_ms,
        "compute_ms": compute_ms,
        "comm_ms": comm_ms,
        "attn_proj_ms": ATTN_PROJ_CONSTANT_MS,
        "attn_core_ms": attn_core_ms,
        "moe_compute_ms": MOE_COMPUTE_CONSTANT_MS,
    }


def compute_total_times(seq_len: int) -> dict:
    """Total time across all layers for one decode step."""
    per_layer = compute_times_per_layer(seq_len)
    return {
        "seq_len": seq_len,
        "kv_load_ms": per_layer["kv_load_ms"] * NUM_LAYERS,
        "compute_ms": per_layer["compute_ms"] * NUM_LAYERS,
        "comm_ms": per_layer["comm_ms"] * NUM_LAYERS,
        "kv_load_per_layer_ms": per_layer["kv_load_ms"],
        "compute_per_layer_ms": per_layer["compute_ms"],
        "comm_per_layer_ms": per_layer["comm_ms"],
        "attn_proj_per_layer_ms": per_layer["attn_proj_ms"],
        "attn_core_per_layer_ms": per_layer["attn_core_ms"],
        "moe_per_layer_ms": per_layer["moe_compute_ms"],
    }


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

results = [compute_total_times(s) for s in SEQ_LENS]

# ── Save results ──────────────────────────────────────────────────────
summary = {
    "description": "DeepSeek-V3 MLA sequence length sweep: KV load vs compute vs comm",
    "model": "deepseek-ai/DeepSeek-V3",
    "attention_type": "MLA",
    "device": "A100 (PCIe Gen4, 31.5 GB/s)",
    "hbm_bandwidth": f"{HBM_BW_GBS} GB/s",
    "pcie_bandwidth": f"{PCIE_BW_GBS} GB/s (effective: {PCIE_BW_GBS * BW_EFFICIENCY} GB/s)",
    "parallelism": f"TP={TP}, EP={EP}",
    "batch_size": 1,
    "kv_bytes_per_token_per_layer": KV_BYTES_PER_TOKEN,
    "num_layers": NUM_LAYERS,
    "sweep_results": results,
}

with open(os.path.join(RESULTS_DIR, "seqlen_sweep_results.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

# ── Print table ───────────────────────────────────────────────────────
print("=" * 95)
print("DeepSeek-V3 MLA Sequence Length Sweep (BS=1, A100 PCIe Gen4)")
print("=" * 95)
print(f"{'Seq Len':>10s}  {'KV Load (ms)':>14s}  {'Compute (ms)':>14s}  {'Comm (ms)':>12s}  "
      f"{'KV/Compute':>11s}  {'KV/Total':>9s}")
print("-" * 95)

for r in results:
    total = r["kv_load_ms"] + r["compute_ms"] + r["comm_ms"]
    seq_label = f"{r['seq_len'] / 1024:.0f}K" if r["seq_len"] < 1024 * 1024 else f"{r['seq_len'] / (1024*1024):.0f}M"
    print(f"{seq_label:>10s}  {r['kv_load_ms']:>14.1f}  {r['compute_ms']:>14.1f}  "
          f"{r['comm_ms']:>12.1f}  {r['kv_load_ms']/r['compute_ms']:>10.1f}x  "
          f"{r['kv_load_ms']/total*100:>8.1f}%")

print("-" * 95)
print(f"\nKV cache per token per layer: {KV_BYTES_PER_TOKEN} bytes "
      f"(MLA: kv_lora_rank={KV_LORA_RANK} + qk_rope_dim={QK_ROPE_HEAD_DIM}, FP16)")
print(f"PCIe effective BW: {PCIE_BW_GBS * BW_EFFICIENCY:.1f} GB/s | "
      f"HBM effective BW: {HBM_BW_GBS * BW_EFFICIENCY:.1f} GB/s")

# ── Plot: grouped bar chart ──────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
})

C_KV = "#3498db"       # Blue for KV cache load
C_COMPUTE = "#2ecc71"  # Green for compute
C_COMM = "#e67e22"     # Orange for communication

fig, ax = plt.subplots(figsize=(14, 7))

seq_labels = []
for s in SEQ_LENS:
    if s < 1024 * 1024:
        seq_labels.append(f"{s // 1024}K")
    else:
        seq_labels.append(f"{s // (1024 * 1024)}M")

x = np.arange(len(SEQ_LENS))
bar_width = 0.25

kv_times = [r["kv_load_ms"] for r in results]
compute_times = [r["compute_ms"] for r in results]
comm_times = [r["comm_ms"] for r in results]

bars_kv = ax.bar(x - bar_width, kv_times, bar_width,
                 label="KV Cache Load (PCIe)", color=C_KV,
                 edgecolor="black", linewidth=0.5, alpha=0.85)
bars_compute = ax.bar(x, compute_times, bar_width,
                      label="Compute (Attn + MoE)", color=C_COMPUTE,
                      edgecolor="black", linewidth=0.5, alpha=0.85)
bars_comm = ax.bar(x + bar_width, comm_times, bar_width,
                   label="Communication (TP + EP)", color=C_COMM,
                   edgecolor="black", linewidth=0.5, alpha=0.85)

ax.set_yscale("log")
ax.set_xlabel("Sequence Length (tokens)")
ax.set_ylabel("Time per Decode Step (ms, log scale)")
ax.set_title(
    "DeepSeek-V3 MLA: Sequence Length Sweep\n"
    "Single Decode Step Timing Breakdown (BS=1, A100 PCIe Gen4, TP=8, EP=8)",
    fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels(seq_labels, fontsize=11)
ax.legend(loc="upper left", fontsize=11)
ax.grid(axis="y", alpha=0.3, which="both")

# Add value labels on top of each bar
for bars in [bars_kv, bars_compute, bars_comm]:
    for bar in bars:
        height = bar.get_height()
        if height >= 1000:
            label = f"{height / 1000:.1f}s"
        elif height >= 1:
            label = f"{height:.0f}"
        else:
            label = f"{height:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 1.15,
            label,
            ha="center", va="bottom", fontsize=7, rotation=45,
        )

# Add annotation about KV load dominance
ax.annotate(
    "KV cache load over PCIe\ndominates at all sequence lengths\n"
    f"(MLA compressed KV: {KV_BYTES_PER_TOKEN} B/token/layer)",
    xy=(4, kv_times[4]),
    xytext=(5.5, kv_times[2] * 0.3),
    fontsize=9, ha="center", color="#2c3e50",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf2f8", edgecolor="#3498db"),
    arrowprops=dict(arrowstyle="->", color="#3498db", lw=1.5),
)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "seqlen_sweep_bar_chart.png")
plt.savefig(plot_path, bbox_inches="tight")
plt.close()

print(f"\nPlot saved to {plot_path}")
print(f"Results saved to {RESULTS_DIR}/")
